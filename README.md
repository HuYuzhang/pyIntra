# pyIntra
Intra analyze with python

## h5.py
* 用于对hdf5文件进行分析，目前反正只有读取hdf5文件的功能，但是h5py确实用起来很方便(

## apply_mode.py
* 根据训练得到的模型生成预测图像，将它和原图共同保存下来
---
# 其他
### 11.17
* 之前得到的预测图像非常奇怪，觉得可能是reshape出了问题，检查了很久，发现是被MATLAB坑到了。MATLAB是
列优先的，所以reshape会出现问题，必须在这里进行reshape的时候指定order='F'保证列优先。
* 另外发现了很多非常没有意义的图片，整张图片几乎都是没有任何变化的。所以之后应该要考虑对于方差小于或者
大于阈值的数据进行过滤（太复杂可能也学不出来？）
### 11.18
* 今天加入了预测结果的PSNR和SSIM测试代码，想要检查一下分模式训练是否真的是有效果的，目前每个模式只测试了1000个输入，结果如下，
暂时就只放PSNR的结果好了（
#### 以200为方差阈值过滤掉无效输入的结果

| data_type\model_type    |dc             | planar           | angle     |
| ------------- |:-------------:| -----:           | -----:    |
| dc            | 19.439        | 19.477           | 19.363    |
|planar         | 19.781        | 19.725           | 19.663    |
|angle          | 19.652        | 19.630           | 19.508    |

#### 不考虑方差统一测试的结果
| data_type\model_type    |dc             | planar           | angle     |
| ------------- |:-------------:| -----:           | -----:    |
| dc            | 21.258        | 21.025           | 20.623    |
|planar         | 24.850        | 24.111           | 23.705    |
|angle          | 21.445        | 21.163           | 20.778    |

* 结果让人很无语，因为基本上每个数据的最佳表现并不总是（甚至很少是）在专门针对于这个模式下进行训练所得到的模型啊摔
* 于是打算要进行一下过滤，将方差太小的块直接扔掉，因为visualize之后看到很多的几乎纯色图，应该没什么用处
---
#### 接下来统计了一下每一个图片的方差，为了之后过滤数据做准备，这里稍微记录一下每一种预测模式的方差情况
| dc       | planar    | angle    |
| ------------- |:-------------:| -----:           |
|971.519   | 794.746   | 1074.991 |
* emmm看来方差还是比较大的，但是除了下界之外是否也需要设置一个上界呢？
