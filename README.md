# Alpha - Mind

<table>
<tr>
  <td>Python version</td>
  <td><img src="https://img.shields.io/badge/python-3.6-blue.svg"/> </td>
  </tr>
<tr>
<tr>
  <td>Build Status</td>
  <td>
    <img src="https://travis-ci.org/alpha-miner/alpha-mind.svg" alt="travis build status" />
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td><img src="https://coveralls.io/repos/github/alpha-miner/alpha-mind/badge.svg?branch=master" alt="coverage" /></td>
</tr>
</table>

**Alpha - Mind** 是基于 **Python** 开发的股票多因子研究框架。

## 依赖

该项目主要有两个主要的github外部依赖：

* [portfolio - optimizer](https://github.com/alpha-miner/portfolio-optimizer)：该项目是相同作者编写的用于资产组合配置的优化器工具包；

* [xgboost](https://github.com/dmlc/xgboost)： 该项目是alpha - mind中一些机器模型的基础库。

这两个库都已经使用git子模块的方式打包到alpha-mind代码库中。

工具依赖包括：

* [cmake](https://cmake.org/)

* [Visual Studio 2015](https://visualstudio.microsoft.com)（仅Windows依赖，Visual Studio 2015以上应该也可以工作，但未做测试）

* [gfortran]()（仅Linux依赖）

在Linux上（例如：Ubuntu）可以使用如下指令完成依赖的安装：

```bash
$ sudo apt-get install git cmake build-essential gfortran -y
```

## 功能

alpha - mind 提供了多因子研究中常用的工具链，包括：

* 数据清洗
* alpha 模型
* 风险模型
* 组合优化
* 执行器

所有的模块都设计了完整的测试用例以尽可能保证正确性。同时，所有的数值模型开发中都对性能给予了足够高的关注，参考了优秀的第三方工具以保证性能：

* numpy
* numba
* cvxopt
* cvxpy
* pandas
* scipy

同时还依赖于一个工具包
* [Finance-Python](https://github.com/alpha-miner/Finance-Python)

## 编译依赖

* Windows

  在Windows上完整安装，需要有C++编译器(例如msvc)
  
  具体可按照如下流程配置(以VS2015为例)：
  
  1. 安装VS2015 社区版，在微软官网可以免费下载。 
  2. 安装CMake, 可以从[官网](https://cmake.org/download/)下载二进制安装文件，如"Windows win64-x64 ZIP"，解压缩后环境变量的设置可以参见[此文](https://blog.csdn.net/liyuebit/article/details/77092723)
     
     - 可以按照文中的例子，尝试使用如下CMake命令编译一个HelloWorld项目。
     ```bash
     cmake -G "Visual Studio 14 2015 Win64"
     ```
     - 将MSBuild的路径(默认是"C:\Program Files (x86)\MSBuild\14.0\Bin"")加入环境变量中。
  
  3. 在项目子目录"\alphamind\pfopt"下使用如下命令进行更新，确保所需文件都已经拷贝到本地。
     ```
     git submodule init
     git submodule update
     ```
     
  4. 在项目根目录下双击批处理文件"build_windows_dependencies.bat"或者通过命令行执行	
     ```bash
     build_windows_dependencies.bat
     ```
     随后一系列依赖项目会自动编译。可能有若干警告，但没有错误。

* Linux

  在linux上，需要c++编译器（例如g++）以及fortran编译器（例如gfortran):
    
    ```bash
    build_linux_dependencies.sh
    ```

## 安装

安装需要直接clone或者下载源代码安装，具体流程为：

克隆项目到本地
```
git clone https://github.com/alpha-miner/alpha-mind.git
cd alpha-mind
git submodule init
git submodule update
cd alphamind/pfopt
git submodule init
git submodule update
cd ../..
```

### SOURCE

1. 参照上节内容，编译好依赖的子项目。

2. 回到项目的根目录下运行：

```python
python setup.py install
```

### Docker

1. `docker build -t alpha-mind:latest -f Dockerfile .`

2. `docker run -it -p 8080:8080 --name alpha-mind alpha-mind`

#### 提示

环境变量的配置在`./entrypoint.sh`中，包括：

* `DB_VENDOR`: 如果使用mysql，请设置为`rl`;
* `DB_URI`: 数据库的连接串。

* *注意事项*: 
1. 在Linux系统上,请确保gcc版本大于4.8;
2. 在libs下面提供了依赖的一些库的二进制文件。linux版本的是在一台具有两个intel cpu的docker虚机上面编译完成的。如果需要实现最佳的性能，建议用户在目标机器上编译相关依赖的库。依赖的库源码地址：[portfolio-optimizer](https://github.com/alpha-miner/portfolio-optimizer)

## 数据源

我们在工具包中也提供了一个数据源的参考实现。这个数据源的参考实现来自于``通联数据``提供的股票因子数据和风险模型数据等，具体细节可以参考：[优矿](https://uqer.io)。

该数据源使用RMDBS，供参考的数据库可以是Postgresql。在工具包中我们已经提供了命令行工具，帮助用户一键配置好数据库。步骤如下：

下面的步骤以Ubuntun上Postgresql为例子：

* 安装数据库软件
  
  请前往[PostgreSQL官网](https://www.postgresql.org/)，根据指导，下载安装PostgreSQL数据库。

* 新建数据库

  在安装完成的数据库中新建``Database``，例如名字：``alpha``。**注意这个数据需要使用``utf8``作为编码**。

* 一键配置数据库

  在命令行中运行：

  ```bash
  alphadmind initdb --url postgresql+psycopg2://user:pwd@host/alpha
  ```

  其中：

  * ``user``：数据库用户名
  * ``pwd``：用户密码
  * ``host``：数据库服务器地址

  如果成功，会有类似的输出：

  ```
  2017-06-29 14:48:36,678 - ALPHA_MIND - INFO - DB: postgresql+psycopg2://user:pwd@host/alpha
  2017-06-29 14:48:37,515 - ALPHA_MIND - INFO - DB: initialization finished.
  ```

* Windows

  对于Windows使用者，命令行工具alphamind并不能直接使用，这个时候可以使用变通的办法，进入源码alphamind/bin目录下：

  ```bash
  python alphadmind initdb --url postgresql+psycopg2://user:pwd@host/alpha
  ```
  
  可以达到一样的效果。
  
* 数据库更新

  在目录``scripts`` 下有[airflow]()脚本文件``update_uqer_data.py``可以用来做每天的数据更新。使用之前除了要配置好airflow服务器之外，需要更新脚本中以下两行：

  ```
  _ = uqer.Client(token='')
  engine = sqlalchemy.create_engine('')
  ```

  其中token需要填入有效的通联数据认证信息；engine需要填入上面指定的数据库地址。

