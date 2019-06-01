import cv2

ix, iy = -1, -1

# mouse callback function
def getxy(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        #ix, iy = x, y
        #print(x, y)
        color = list(img[y][x])
        color = color[::-1]
        print("Color in (r,g,b) at ", (x,y) ,":",color)


img = cv2.imread(r"D:\Intern\test images\test4.jpg")
cv2.namedWindow('image')
cv2.setMouseCallback('image', getxy)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()