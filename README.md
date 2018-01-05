# Alpha - Mind

<table>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/wegamekinglc/alpha-mind">
    <img src="https://travis-ci.org/wegamekinglc/alpha-mind.svg?branch=master" alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Coverage</td>
  <td><img src="https://coveralls.io/repos/github/wegamekinglc/alpha-mind/badge.svg?branch=master" alt="coverage" /></td>
</tr>
</table>

**Alpha - Mind** 是基于 **Python** 开发的股票多因子研究框架。

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

## 编译依赖

* Windows

  在Windows上完整安装，需要有C++编译器(例如msvc）:

    ```bash
    build_windows_dependencies.bat
    ```

* Linux

    在linux上，需要c++编译器（例如g++）以及fortran编译器（例如gfortran)

    ```bash
    build_linux_dependencies.sh
```

## 安装

alpha - mind 的安装极其简单，在编译完成依赖之后，运行：

```python
python setup.py install
```

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

