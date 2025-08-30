# 模型下载归档与索引（代码结构与流程分析）

本文档梳理本仓库中与“模型下载归档（blobs 与 manifests）”和“本地索引（已下载模型与层的索引/清单）”相关的核心代码、数据结构与执行流程，便于后续维护与扩展。

## 目录与命名概览

本地模型缓存以两类文件为主：
- manifests：每个模型的清单（manifest，JSON），记录该模型由哪些层（layers）组成，以及配置层（config）。
- blobs：每个层（layer）的实际二进制内容，按 sha256 摘要命名。

关键路径工具：
- GetManifestPath()：返回本地清单根目录路径，并确保目录存在。
- GetBlobsPath(digest)：返回某个 digest 对应的 blob 文件路径；当 digest 为空字符串时返回 blobs 目录，并确保目录存在。含严格的 sha256 格式校验（不合规的 blob 会在清理时删除）。

相关代码位置：
- server/modelpath.go：ModelPath 解析、GetManifestPath、GetBlobsPath 等。
- server/manifest.go：Manifest 结构体、读写与枚举（索引）函数。
- server/images.go：PullModel 拉取流程、写入 manifest、校验与清理；PushModel 也在此（上传相关）。
- server/download.go：分块/并发下载、断点续传、下载去重与进度跟踪。
- server/layer.go：单层文件的打开与按引用计数方式删除。

## 核心数据结构

- Manifest（server/manifest.go）
  - 字段：SchemaVersion, MediaType, Config（Layer）, Layers（[]Layer）
  - 方法：Size(), Remove(), RemoveLayers()
- Layer（server/layer.go）
  - 关键字段：Digest, MediaType, Size
  - 方法：Open(), Remove()
- ModelPath（server/modelpath.go）
  - 负责解析形如 [scheme://]registry/namespace/repository:tag 的模型标识，并提供 BaseURL、GetManifestPath、GetBlobsPath 等。
- registryOptions（server/images.go）
  - 拉取/上传请求的认证、insecure 开关等选项。

## 本地索引（已下载模型/层的索引）

索引的来源是“manifests 目录中的清单文件集合”。
- Manifests(continueOnError bool)（server/manifest.go）：
  - 遍历 manifests 根目录下的层级文件路径，解析为 model.Name，再使用 ParseNamedManifest 读取并校验 JSON 清单，最终返回 map[model.Name]*Manifest。
  - 该函数的返回值是“本地所有模型清单的索引”，被清理/删除等流程复用。
- ParseNamedManifest/WriteManifest（server/manifest.go）：分别用于读取/写入清单，写入时会创建必要目录。

## 拉取流程（下载与归档）

入口：PullModel(ctx, name, regOpts, progressFn)（server/images.go）

1) 解析 ModelPath，尝试读取本地旧 manifest，用于构建 deleteMap（候选删除的旧层集合）。
2) 拉取远端 manifest：pullModelManifest(ctx, mp, regOpts)。
3) 组装 layers 列表（包含 Manifest.Layers 与 Config），逐层下载：
   - downloadBlob(ctx, downloadOpts)（server/download.go）：
     - 先调用 GetBlobsPath(digest) 查找本地缓存；若已存在则直接命中（cache hit），上报完成进度并跳过下载。
     - 否则使用 blobDownloadManager 确保同一 digest 只会有一个并发下载（多处请求共享进度）。
     - 首次下载会创建 blobDownload，调用 Prepare 与 Run 启动下载。
4) 校验阶段：verifyBlob(digest)（server/images.go）
   - 对非 cache hit 的层文件计算 sha256，与 digest 比对；不一致则删除该 blob 并返回错误。
5) 写入 manifest：将远端 manifest JSON 写入本地 manifests（确保目录存在）。
6) 清理未使用层：deleteUnusedLayers(deleteMap)（server/images.go）
   - 读取当前所有本地清单（Manifests），将仍被任何清单引用的 digest 从 deleteMap 删除；最终对残余的 digest 调用 os.Remove 删除 blob 文件。
7) 完成：progress 状态为 success。

进度上报：
- 状态包括 "pulling manifest"、"pulling <digest>"、"verifying sha256 digest"、"writing manifest"、"removing unused layers"、"success" 等，便于前端/CLI 显示拉取过程。

## 下载实现细节（分块、并发、断点续传）

核心类型：blobDownload（server/download.go）
- 管理单个 digest 的下载，支持：
  - 分块并发（默认 16 片，块大小在 [100MB, 1000MB] 范围内）
  - 断点续传：使用 <目标文件名>-partial 和 <目标文件名>-partial-<i>（JSON）存储整体与分块的中间状态
  - 进度共享：通过 blobDownloadManager（sync.Map）避免重复下载，同 digest 调用者共享进度

关键流程：
- Prepare(ctx, url, opts)
  - 扫描已存在的 "-partial-*" 分块状态文件，恢复 Parts 信息，计算 Total/Completed
  - 若首次下载则创建新的分块元信息文件
- Run(ctx, directURL, opts)
  - 获取 directURL（跟随首次 302 等跳转但限制不同 host 的重定向），
  - 使用 errgroup 并发拉取各分块：
    - downloadChunk：设置 Range: bytes=start-end，按分块写入到目标文件的偏移位置（io.NewOffsetWriter），并通过 io.TeeReader 统计进度
    - 每个分块写入后更新对应的 "-partial-<i>" JSON 文件（writePart）
    - 带指数退避与“分块卡顿”检测（30s 未更新认为卡顿，返回重试）
  - 所有分块完成后，关闭文件、删除所有 "-partial-<i>"，并将 "-partial" 原子重命名为最终 blob 路径

失败与重试策略：
- 分块最多重试 6 次（指数退避 1s, 2s, 4s, ...），对 ENOSPC 或 context 取消将立即中止。
- 若分块长时间无进度，会触发 errPartStalled 并重试该分块。

## 清理与引用安全

- deleteUnusedLayers（server/images.go）：结合 Manifests() 索引，删除所有“未被任何 manifest 引用”的 blob。
- PruneLayers（server/images.go）：扫描 blobs 目录，剔除非法命名（非 sha256 格式）的 blob，并调用 deleteUnusedLayers 做进一步清理。
- Layer.Remove（server/layer.go）：
  - 在删除单个层前，会加载现有所有 Manifests，若该 digest 仍被任何清单引用则放弃删除，避免误删。

## 小结

- 本地“索引”即 manifests 目录下的所有清单集合，Manifests() 提供统一读取接口，支撑清理与引用检查。
- 下载采用“按层（digest）”的去重、分块、断点续传与强一致校验（sha256），并在完成后写入 manifest 与清理旧层，形成完整的“下载归档 + 索引维护”闭环。

## 主要文件一览（便于跳转）
- server/modelpath.go：ModelPath、GetManifestPath、GetBlobsPath
- server/manifest.go：Manifest 结构、ParseNamedManifest、WriteManifest、Manifests
- server/images.go：PullModel、deleteUnusedLayers、PruneLayers、verifyBlob
- server/download.go：blobDownload 及其 Prepare/Run/downloadChunk 等
- server/layer.go：Layer.Open/Remove