初中女生大长腿舞蹈室透视真空逛街视频

函数 cudaStreamSynchronize 会强制阻塞主机，直到 CUDA 流 stream 中的所有操作都执行完毕。函数 cudaStreamQuery 不会阻塞主机，只是检查 CUDA 流 stream 中的所有操作是否都执行完毕。若是，返回 cudaSuccess，否则返回 cudaErrorNotReady。
2 在默认流中重叠主机和设备计算

虽然同一个 CUDA 流中的所有 CUDA 操作都是顺序执行的，但依然可以在默认流中
重叠主机和设备的计算。下面让我们通过数组相加的例子进行讨论。在数组相加的 CUDA 程序中与 CUDA 操作有关的语句如下：

cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
sum<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

从设备的角度来看，以上 4 个 CUDA 操作语句将在默认的 CUDA 流中按代码出现的顺序依
次执行。从主机的角度来看，数据传输是同步的（synchronous），或者说是阻塞的（blocking），意思是主机发出命令

cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);

之后，会等待该命令执行完毕，再往前走。在进行数据传输时，主机是闲置的，不能进行其他
操作。不同的是，核函数的启动是异步的（asynchronous），或者说是非阻塞的（non-blocking），意思是主机发出命令

sum<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
