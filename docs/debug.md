* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=2, address=0x9c1c00000)
    frame #0: 0x0000000192b1a3c0 libsystem_platform.dylib`_platform_memmove + 96
libsystem_platform.dylib`_platform_memmove:
->  0x192b1a3c0 <+96>:  ldnp   q0, q1, [x1]
    0x192b1a3c4 <+100>: add    x1, x1, #0x20
    0x192b1a3c8 <+104>: subs   x2, x2, #0x20
    0x192b1a3cc <+108>: b.hi   0x192b1a3b8    ; <+88>
Target 2: (module_integration_test) stopped.
(lldb) bt
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=2, address=0x9c1c00000)
  * frame #0: 0x0000000192b1a3c0 libsystem_platform.dylib`_platform_memmove + 96
    frame #1: 0x000000010017ea48 module_integration_test`duorou::ml::nn::MultiHeadAttention::forward(this=0x00000009b5123a00, ctx=0x00000009b4c08100, query=0x000000016fdfbbf0, key=0x000000016fdfbb00, value=0x000000016fdfbac8, cache=0x00000009b4c18d80, mask=0x000000016fdfba90) at attention.cpp:140:17
    frame #2: 0x0000000100051e74 module_integration_test`duorou::model::SelfAttention::forward(this=0x00000009b511d960, ctx=0x00000009b4c08100, input=size=65536, attentionMask=size=0, cache=0x00000009b4c18d80) at qwen_text_model.cpp:208:34
    frame #3: 0x0000000100053434 module_integration_test`duorou::model::TransformerLayer::forward(this=0x00000009b5012440, ctx=0x00000009b4c08100, input=size=65536, attentionMask=size=0, cache=0x00000009b4c18d80) at qwen_text_model.cpp:348:24
    frame #4: 0x0000000100057edc module_integration_test`duorou::model::QwenTextModel::forward(this=0x0000000101ab8200, ctx=0x00000009b4c08100, inputIds=0x000000016fdfcfc8, cache=0x00000009b4c18d80) at qwen_text_model.cpp:879:33
    frame #5: 0x0000000100074184 module_integration_test`duorou::model::QwenMultimodalModel::forward(this=0x0000000101ab7700, ctx=0x00000009b4c08100, inputIds=0x000000016fdfcfc8, pixelValues=size=0, cache=0x00000009b4c18d80) at qwen_multimodal_model.cpp:351:37
    frame #6: 0x00000001000ef4a0 module_integration_test`duorou::extensions::ollama::MLInferenceEngine::generateWithInternalForward(this=0x000000016fdfd210, prompt="你好，马上是中秋节了，帮我写一首诗，并翻译为英文。", max_tokens=256, temperature=0.699999988, top_p=0.899999976) at inference_engine.cpp:1311:36
    frame #7: 0x00000001000ee094 module_integration_test`duorou::extensions::ollama::MLInferenceEngine::generateText(this=0x000000016fdfd210, prompt="你好，马上是中秋节了，帮我写一首诗，并翻译为英文。", max_tokens=256, temperature=0.699999988, top_p=0.899999976) at inference_engine.cpp:359:14
    frame #8: 0x000000010000449c module_integration_test`testGGMLInference() at test_module_integration.cpp:244:58 [opt]
    frame #9: 0x0000000100005f18 module_integration_test`main at test_module_integration.cpp:428:10 [opt]
    frame #10: 0x0000000192751d54 dyld`start + 7184
(lldb) quit