# cs240ahw1

Description:    Naive dgemm
Size: 1600    Mflop/s: 231.731    Time: 35.3513
Size: 800    Mflop/s: 1513.55    Time: 0.676554

Description:    Simple blocked dgemm.
Size: 1600    Mflop/s: 3348.57    Time: 2.44642
Size: 800    Mflop/s: 3599.69    Time: 0.284469

Description:    BLAS dgemm.
Size: 1600    Mflop/s: 38032.3    Time: 0.215396
Size: 800    Mflop/s: 39915        Time: 0.0256545

Description:    omp4_v1: blocked dgemm -each handles 1/4 of consecutive block colums.
Size: 1600    Mflop/s: 12500        Time: 0.655359
Size: 800    Mflop/s: 13915.7    Time: 0.073586

-------------------------------------------------------------------------------------
//BLOCK_SIZE = 40
Description:    Naive, three-loop dgemm.
Size: 1600    Mflop/s: 255.657    Time: 32.043
Size: 800    Mflop/s: 1579.99    Time: 0.64810

Description:    Simple blocked dgemm.
Size: 1600    Mflop/s: 3835.1        Time: 2.13606
Size: 800    Mflop/s: 4061.27    Time: 0.252138

Description:    BLAS dgemm.
Size: 1600    Mflop/s: 41945.9    Time: 0.195299
Size: 800    Mflop/s: 43381.1    Time: 0.0236048

Description:    omp4_v1: blocked dgemm -each handles 1/4 of consecutive block colums.
Size: 1600    Mflop/s: 14348.4    Time: 0.570935
Size: 800    Mflop/s: 14727        Time: 0.069532

-------------------------------------------------------------------------------------
//BLOCK_SIZE = 41
Description:    Simple blocked dgemm.
Size: 1600    Mflop/s: 3150.42    Time: 2.60029
Size: 800    Mflop/s: 3735.07    Time: 0.274158

Description:    blocked dgemm omp v3 -- similar to v1, but BLOCK_SIZE = 41
Size: 1600    Mflop/s: 13228.4    Time: 0.619272
Size: 800    Mflop/s: 13693.9    Time: 0.074778

Description:    blocked dgemm omp v4 - parallel for, block schedule
Size: 1600    Mflop/s: 12725.5    Time: 0.643747
Size: 800    Mflop/s: 13153.2    Time: 0.077852

Description:    blocked dgemm pthread.
Size: 1600    Mflop/s: 6423.29    Time: 1.27536

------------------------------------------------------------------------------------
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 13705        Time: 0.074717
Size: 1600    Mflop/s: 13253.6    Time: 0.618097

16 threads
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 33908.1    Time: 0.0301992
Size: 1600    Mflop/s: 43830.9    Time: 0.1869

24 threads
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 24621.2    Time: 0.0415903
Size: 1600    Mflop/s: 35230.9    Time: 0.232523

--------------------------------------------------------------------------------------
4 threads, 4 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 13900.2    Time: 0.073668
Size: 1600    Mflop/s: 13408.7    Time: 0.610946

12 threads, 12 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 34600.4    Time: 0.029595
Size: 1600    Mflop/s: 33483.2    Time: 0.24466

16 threads, 16 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 34539.8    Time: 0.029647
Size: 1600    Mflop/s: 44518.6    Time: 0.184013

20 threads, 20 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 69144.8    Time: 0.0148095
Size: 1600    Mflop/s: 66922.1    Time: 0.122411

21 threads, 21 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 69064.9    Time: 0.0148266
Size: 1600    Mflop/s: 64748.1    Time: 0.126521

22 threads, 22 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 63619.8    Time: 0.0160956
Size: 1600    Mflop/s: 65021        Time: 0.12599

23 threads, 23 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 32080.2    Time: 0.03192
Size: 1600    Mflop/s: 65037    Time: 0.125959

24 threads, 24 task per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 27357    Time: 0.037431
Size: 1600    Mflop/s: 32904    Time: 0.248967

24 threads, 24 tasks per node
Description:    blocked dgemm omp v3.
Size: 800    Mflop/s: 32377.4    Time: 0.031627
Size: 1600    Mflop/s: 29950.4    Time: 0.273519


