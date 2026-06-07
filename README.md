# 小鸟游韦布/weibu AI大模型
小鸟游韦布/reyoweibu/weibu是由Qwen2.5:3b模型微调而成
通过.gguf文件轻松使用
独家语气 彰显特色
---
## 部署教程
1.安装ollama
首先，你要确定你用的是啥系统，默认以linux为例，粘贴以下命令一键安装（windows和macos请移步官网下载安装包）
```
curl -fsSL https://ollama.com/install.sh | sh
```
2.下载需要的文件并进入目录
这里推荐大家使用git一键clone
```
git clone https://github.com/buliuming2/weibuai.git
```
接下来你要进入你部署的目录，把模型下载
```
cd weibuai
```
3.下载模型权重
Linux/Unix/macOS:
```
wget “https://modelscope.cn/models/keyoweb/weibu/file/view/master/weibu1.0.1_3b.gguf?status=2”
```
Windows或者命令运行失败请把链接粘贴至浏览器/下载器/PCL2（这个支持下载）
4.部署模型
首先你要将mf文件编译到模型里面
```
ollama create weibu1.0.1_3b -f ./weibuai.mf
```
然后运行
```
ollama run weibu-qwen3b
```
这样，你的模型就部署成功。
## 相关链接
[ollama官网](https://ollama.com/)
[模型权重地址](https://modelscope.cn/models/keyoweb/weibu/files)
