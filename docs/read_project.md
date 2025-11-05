## 主要文件的分析
#### src/extensions/ollama/ollama_model_manager.h
它是一个非常重要的文件，它用来组织管理ollama模型，ollama模型应该是采用一种对象化的管理模式，它由
``manisets``和``blobs``组成，``manisets``通过子文件夹的方式来区分这个模型是注册在ollama的还是
第三方的，如果是ollama的将会在registry.ollama.ai文件夹下的library文件下,以模型文件名建立对应的
文件夹。最后在这个文件中存放模型存放的实际地址的文件，该文件通过json格式保存，该文件一般以版本和
量化信息为文件名,如果打开实际文件内容如下(这里是我们下载的qwen2.5vl)：
```json
{
    "schemaVersion": 2,
    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
    "config": {
        "mediaType": "application/vnd.docker.container.image.v1+json",
        "digest": "sha256:83b9da835d9f13632a97e550cc8fd02ff7f39b88a843f0a8923330890682977a",
        "size": 567
    },
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025",
            "size": 5969233408
        },
        {
            "mediaType": "application/vnd.ollama.image.template",
            "digest": "sha256:a242d8dfdc8f8c2b0586ee85fba70adb408fb633aba2836fe1b05f2c46631474",
            "size": 487
        },
        {
            "mediaType": "application/vnd.ollama.image.system",
            "digest": "sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
            "size": 28
        },
        {
            "mediaType": "application/vnd.ollama.image.license",
            "digest": "sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
            "size": 11343
        },
        {
            "mediaType": "application/vnd.ollama.image.params",
            "digest": "sha256:52d2a7aa3a380c606bd1cd3d6f777a9c65a1c77c2e0cb091eed2968a5ef04dc3",
            "size": 23
        }
    ]
}
```
可以看到文件中主要包含了基础的配置信息，以及模型层次的信息，各个层（layers）的含义： 
1. application/vnd.ollama.image.model 
作用：这是模型的核心权重文件  大小：5,969,233,408 字节（约 5.6GB）  
说明：包含了神经网络的实际参数和权重数据   
2. application/vnd.ollama.image.template 
作用：模型的提示词模板  大小：487 字节  
说明：定义了如何格式化输入提示词，比如聊天格式、系统提示词的结构等   
3. application/vnd.ollama.image.system 
作用：系统级配置信息  大小：28 字节  
说明：包含模型的系统级设置和默认行为配置   
4. application/vnd.ollama.image.license 
作用：许可证信息  大小：11,343 字节  
说明：模型的使用许可证条款和法律信息   
5. application/vnd.ollama.image.params 
作用：模型参数配置  大小：23 字节  
说明：模型的运行时参数，如温度、top-p 等推理参数的默认值   
整体架构意义： 这种分层设计的好处是： 
模块化管理：不同类型的数据分开存储  
增量更新：可以单独更新某个层而不影响其他部分  
存储优化：相同的层可以在不同模型间共享  
版本控制：每个层都有独立的 SHA256 哈希值用于完整性验证

我们再来看看blobs文件夹：
目录: ~/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5vl

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2025/9/15     10:39             23 sha256-52d2a7aa3a380c606bd1cd3d6f777a9c65a1c77c2e0cb091eed2968a5ef04dc3  -> application/vnd.ollama.image.params
-a----         2025/9/15     10:39             28 sha256-75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de  -> application/vnd.ollama.image.system
-a----         2025/9/15     10:39          11343 sha256-832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  -> application/vnd.ollama.image.license
-a----         2025/9/15     10:39            567 sha256-83b9da835d9f13632a97e550cc8fd02ff7f39b88a843f0a8923330890682977a  -> application/vnd.docker.container.image.v1+json
-a----         2025/9/15     10:39            487 sha256-a242d8dfdc8f8c2b0586ee85fba70adb408fb633aba2836fe1b05f2c46631474  -> application/vnd.ollama.image.template
-a----         2025/9/15     10:39     5969233408 sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025  -> application/vnd.ollama.image.model