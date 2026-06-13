import time
t0 = time.time()
for i in range(30):
    model(frame)
print(time.time() - t0)