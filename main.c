#include <liburing/io_uring.h>
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl_platform.h>
#include <CL/cl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <liburing.h>
#include <assert.h>

double
elapsed(struct timespec start, struct timespec end)
{
  return (1.0e9 * (double)(end.tv_sec - start.tv_sec)) + (double)(end.tv_nsec - start.tv_nsec);
}

int
load_file(const char *source_path,const char** content)
{
  struct stat stat_buf;
  char * buf;
  size_t size;
  int fd;
  if(stat(source_path, &stat_buf) == -1) {
    return -1;
  };
  size = stat_buf.st_size;
  buf = (char *)malloc(size);
  fd = open(source_path, O_RDONLY);
  if(read(fd, buf, size) == -1) {
    return -1;
  }
  *content = buf;
  return 0;
}

int
main(int argc, char **argv)
{
  int n, ndigits, cell_size;
  if (1 < argc) {
    n = atoi(argv[1]);
    ndigits = 8 < strlen(argv[1]) ? strlen(argv[1]) : 8;
  } else {
    n = 1024 * 1024 * 1024;
    ndigits = 10;
  }
  cell_size = (ndigits + 1) * 15;

  size_t global_work_size = (n + 14) / 15;
  // preferred work group size
  size_t local_work_size = global_work_size > 256 ? 256 : global_work_size;
  size_t local_mem_size = cell_size * local_work_size;
  // size_t workgroup size = 256;
  size_t batch_size = local_work_size * 1024;
  size_t num_batches = (global_work_size + batch_size - 1) / batch_size;

  // load source file
  const char *source;
  if(load_file("./fizzbuzz.clc", &source)) {
    fprintf(stderr, "failed to load kernel file\n");
    return -1;
  }

  ////////////////////////////////////////////////////////////////
  // setup io_uring
  struct io_uring ring;
  int io_ret;
  io_ret = io_uring_queue_init(num_batches, &ring, 0);
  if (io_ret < 0) {
    fprintf(stderr, "io_uring_queue_init: %d\n", io_ret);
    return -1;
  }

  // 1. Get a platform
  cl_platform_id platform;
  cl_int ret;
  ret = clGetPlatformIDs(1, &platform, NULL);
  if(ret != CL_SUCCESS) {
    fprintf(stderr, "clGetPlatformIDs: %d\n", ret);
    return -1;
  }

  // 2. Find a gpu device.
  cl_device_id device;
  ret = clGetDeviceIDs(
                 platform,
                 CL_DEVICE_TYPE_GPU,
                 1,
                 &device,
                 NULL);
  if(ret != CL_SUCCESS) {
    fprintf(stderr, "clGetDeviceIDs: %d\n", ret);
    return -1;
  }

  // 3. Create a context and command queue on that device.
  cl_context context = clCreateContext(
                                       NULL,
                                       1,
                                       &device,
                                       NULL,
                                       NULL,
                                       NULL);

  cl_command_queue queue = clCreateCommandQueue(
                                                context,
                                                device,
                                                0,
                                                NULL);

  // 4. Perform runtime source compilation, and obtain kernel entry point.
  cl_program program = clCreateProgramWithSource(
                                                 context,
                                                 1,
                                                 &source,
                                                 NULL,
                                                 NULL);
  ret = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", NULL, NULL);
  if(ret != CL_SUCCESS) {
    char buffer[2048];
    size_t length;
    fprintf(stderr, "clBuildProgram: %d\n", ret);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    buffer[length] = '\0';
    fprintf(stderr, "%s", buffer);
    return ret;
  }

  cl_kernel kernel = clCreateKernel(program, "fizzbuzz", NULL);

  // 5. Create a data buffer.
  cl_mem buffer = clCreateBuffer(
                                 context,
                                 CL_MEM_WRITE_ONLY,
                                 cell_size * global_work_size,
                                 NULL,
                                 NULL);
  // 6. Launch the kernel. Let OpenCL pick the local work size.
  ret = clSetKernelArg(kernel, 0, sizeof(buffer), (void *) &buffer);
  if(ret != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg: %d\n", ret);
    return ret;
  }
  ret = clSetKernelArg(kernel, 1, local_mem_size, NULL);
  if(ret != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg: %d\n", ret);
    return ret;
  }
  ret = clSetKernelArg(kernel, 2, sizeof(ndigits), (void *) &ndigits);
  if(ret != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg: %d\n", ret);
    return ret;
  }

  struct iovec *v = calloc(batch_size, sizeof(struct iovec));
  for(size_t offset = 0; offset < global_work_size; offset += batch_size) {
    size_t worksize = (global_work_size - offset) > batch_size? batch_size : (global_work_size - offset);
    size_t num_groups = (worksize + local_work_size - 1) / local_work_size;
    ret = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 1,
                                 &offset,
                                 &worksize,
                                 &local_work_size,
                                 0,
                                 NULL,
                                 NULL);
    if(ret != CL_SUCCESS) {
      fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", ret);
      return ret;
    }

    ret = clFinish(queue);
    if(ret != CL_SUCCESS) {
      fprintf(stderr, "clFinish: %d\n", ret);
      return ret;
    }
    cl_char *ptr;
    ptr = (cl_char *) clEnqueueMapBuffer(
                                         queue,
                                         buffer,
                                         CL_TRUE,
                                         CL_MAP_READ,
                                         cell_size * offset,
                                         cell_size * worksize,
                                         0,
                                         NULL,
                                         NULL,
                                         &ret);
    if (ret != CL_SUCCESS) {
      fprintf(stderr, "clEnqueueMapBuffer: %d\n", ret);
      return ret;
    }

    for(size_t j = 0; j < num_groups; j++) {
      v[j].iov_len = *(size_t *)((char *)(&ptr[j * local_mem_size]));
      v[j].iov_base = &ptr[j * local_mem_size + 8];
    }
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    assert(sqe);
    io_uring_prep_writev(sqe, 1, v, num_groups, 0);
    io_uring_submit(&ring);
  }
  struct io_uring_cqe *cqe;
  for(size_t i = 0; i < num_batches; i++) {
    io_ret = io_uring_wait_cqe(&ring, &cqe);
    if (io_ret) {
      fprintf(stderr, "wait_cqe=%d\n", ret);
      return 1;
    }
    if (cqe->res < 0) {
      fprintf(stderr, "write res=%d\n", cqe->res);
      return 1;
    }
    io_uring_cqe_seen(&ring, cqe);
  }

  /* printf("cell_size: %d\nglobal_work_size: %lu\nlocal_mem_size: %lu\nglobal_mem_size: %lu\ncount: %d\n", */
  /*        cell_size, */
  /*        global_work_size, */
  /*        local_mem_size, */
  /*        global_work_size * cell_size, */
  /*        n); */


  return 0;
}

