## Alpha-Mind 安装指南

### 一、安装环境

#### mysql驱动

`Alpha-Mind`使用过程中调用的因子数据保存在数据库中，因此，`Alpha-Mind`的运行环境需要安装适当的数据库驱动。这里我们选择使用`mysql`。

`mysql`驱动官网下载地址：

https://dev.mysql.com/downloads/

进入官网下载网址后，可以看到以下页面：

![](001.png)

点击`Connector/Python`，进入`Python`驱动下载页面：

![]()![002](002.png)

进入页面后：

- 选择正确的操作系统
- 选择正确的操作系统版本
- 选择合适的版本进行下载

下载并安装数据库驱动

#### 编译器

`Alpha-Mind`依赖的一些包，如`ecos`需要编译安装模式，所以，`Alpha-Mind`运行安装的环境需要有C编译器。

- Windows环境
  请安装Virual Studio 2015以上的版本

- Linux环境

  ```bash
  yum -y install gcc
  yum -y install gcc-c++
  ```

  或者

  ```bash
  apt-get install gcc
  apt-get install gcc-c++
  ```



### 二、安装

在安装环境准备充分后，执行以下命令，开始安装`Alpha-Mind`开发包

```bash
pip install Alpha-Mind
```

如果网络环境不好，可以选择国内的镜像，例如阿里云镜像

```bash
pip install Alpha-Mind -i https://mirrors.aliyun.com/pypi/simple
```

#### 环境变量设置

`Alpha-Mind`安装完成以后，需要正确设置环境变量，才能正常工作：

- **DB_VENDOR**: 数据库类型，这里可以设置为`mysql`：

  `DB_VENDOR=mysql`

- **DB_URI**: 数据库连接配置

  `DB_URI=mysql+mysqldb://rlUser:123456@10.16.50.12:3306/rl?charset=utf8`

- **FACTOR_TABLES**: 因子数据表名称，用`,`分割，例如：

  `FACTOR_TABLES=factor_momentum,factor_power_volume,factor_basic_derivation`