import tensorflow as tf
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3.,3.]])

# 创建另外一个常量op，产生一个2x1矩阵
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)


# 其中默认图。
sess = tf.Session()

# 调用sess的run()方法来执行矩阵乘法
# 上面提到，product代表了矩阵乘法op的输出，传入它是向方法表明，
# 我们希望取回矩阵乘法op的输出。

# 整个执行过程是自动化的，会话负责传递op的所有输入，op通常是并发执行的

# 返回值result是一个numpy ndarrary对象

result = sess.run(product)
print(result)

#任务完成，关闭会话。
sess.close()

# 或者通过with语句
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
