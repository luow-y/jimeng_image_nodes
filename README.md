# 即梦AI图像生成节点（Token版）

## 简介
这是一个ComfyUI自定义节点包，支持即梦AI的文生图和图生图功能。使用Token（sessionid）直接调用，无需配置账号。

## 功能特性
- ✅ 文生图：通过文本提示词生成图片
- ✅ 图生图：基于参考图和提示词生成新图片
- ✅ 支持1-6张参考图
- ✅ 支持多种模型和分辨率
- ✅ 直接使用Token，无需配置文件

## 安装方法

### 方法1：直接复制
1. 将整个 `jimeng_image_nodes` 文件夹复制到ComfyUI的 `custom_nodes` 目录
2. 重启ComfyUI

### 方法2：Git克隆
```bash
cd ComfyUI/custom_nodes
git clone [你的仓库地址] jimeng_image_nodes
cd jimeng_image_nodes
pip install -r requirements.txt
```

## 使用方法

### 获取Token（sessionid）
1. 打开即梦官网：https://jimeng.jianying.com
2. 登录您的账号
3. 按F12打开开发者工具
4. 切换到"应用程序"（Application）标签
5. 在左侧找到"Cookies" > "https://jimeng.jianying.com"
6. 复制 `sessionid` 的值

### 在ComfyUI中使用

#### 文生图
1. 添加 `即梦AI生图（Token版）` 节点
2. 在 `token` 框中粘贴您的sessionid
3. 填写提示词
4. 选择模型、分辨率、比例等参数
5. 运行

#### 图生图
1. 添加 `即梦AI生图（Token版）` 节点
2. 在 `token` 框中粘贴您的sessionid
3. 填写提示词
4. 连接1-6张参考图到 `ref_image_1` 至 `ref_image_6`
5. 运行

## 参数说明

- **prompt**：提示词，描述您想要生成的图像
- **token**：即梦sessionid（必填）
- **model**：选择模型版本
- **resolution**：分辨率（1k/2k/4k）
- **ratio**：图片比例
- **seed**：随机种子（-1为随机）
- **num_images**：生成图片数量（1-4张）
- **ref_image_1-6**：可选的参考图片

## 输出说明

- **images**：生成的图片（张量格式）
- **generation_info**：生成信息文本
- **image_urls**：图片URL列表
- **history_id**：历史记录ID

## 积分消耗

- **文生图**：1积分/张
- **图生图**：2积分/张

## 常见问题

### Q: Token无效
A: 请检查：
- sessionid是否正确复制
- sessionid是否已过期（重新登录获取）
- 网络连接是否正常

### Q: 积分不足
A: 
- 文生图需要1积分
- 图生图需要2积分
- 请到即梦官网充值

### Q: 生成失败
A: 请检查：
- 提示词是否包含敏感词
- 参考图格式是否正确
- 网络连接是否稳定

## 技术支持
- 查看ComfyUI控制台日志
- 访问项目GitHub Issues

## 许可证
请查看 LICENSE 文件

## 版本历史
- v1.0.0 - 初始版本，支持文生图和图生图



