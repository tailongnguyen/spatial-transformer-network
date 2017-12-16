import matplotlib.pyplot as plt 
import cv2
im = cv2.imread('../images/0.png', 0)
fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax = fig.add_subplot(121)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title 1')
plt.imshow(im)
ax = fig.add_subplot(122)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title 2')
plt.imshow(im)

plt.show()