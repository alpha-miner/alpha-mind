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

所有的模块都设计了完整的测试用例以尽可能保证正确性。同时，所有的数值模型开发中都对性能给予了足够高的关注，参考了优秀的第三方工具以保证性能：

* numpy
* numba
* cvxopt
* cvxpy
* pandas
* scipy

## 安装

alpha - mind 的安装及其简单，只需要在下载源码之后，运行：

```python
python setup.py install
```

## 数据源

为了尽可能的保证多因子研究工具的适用性，几乎所有的函数功能都不依赖一个具体的外部数据源。但是为了整个体系的完整性，我们在工具包中也提供了一个数据源的参考实现。这个数据源的参考实现来自于``通联数据``提供的股票因子数据和风险模型数据等，具体细节可以参考：[优矿](https://uqer.io)。

该数据源使用RMDBS，供参考的数据库可以是，例如：MySQL或者SQL server。在工具包中我们已经提供了命令行工具，帮助用户一键配置好数据库。步骤如下：

下面的步骤以Ubuntun上MySQL为例子，Windows服务器以及SQL server等其他RMDBS的配置类似。

* 安装数据库软件

  ```bash
  suod apt-get install mysql-server
  ```

* 新建数据库

  在安装完成的数据库中新建``Database``，例如名字：``multi_factor``。**注意这个数据需要使用``utf8``作为编码**。

* 一键配置数据库

  在命令行中运行：

  ```bash
  alphadmind initdb --url mysql+mysqldb://user:pwd@host/multi_factor?charset=utf8
  ```

  其中：

  * ``user``：数据库用户名
  * ``pwd``：用户密码
  * ``host``：数据库服务器地址

  如果成功，会有类似的输出：

  ```
  2017-06-29 14:48:36,678 - ALPHA_MIND - INFO - DB: mysql+mysqldb://user:pwd@host/multi_factor?charset=utf8
  2017-06-29 14:48:37,515 - ALPHA_MIND - INFO - DB initialization finished.
  ```

* Windows

  对于Windows使用者，命令行工具alphamind并不能直接使用，这个时候可以使用变通的办法，进入源码alphamind/bin目录下：

  ```bash
  python alphadmind initdb --url mysql+mysqldb://user:pwd@host/multi_factor?charset=utf8
  ```
  
  可以达到一样的效果。

