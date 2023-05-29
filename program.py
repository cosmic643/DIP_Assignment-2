import cv2
import numpy as np
mat = cv2.imread("input_image.jpg")
#print(mat)
if mat is None:
    print("Error: Image cannot be loaded")
    exit(0)
else:
    print("Image read successfully")

bgr_img = mat
mat = cv2.cvtColor(mat,cv2.COLOR_BGR2HSV)

height,breadth,channel = mat.shape
img = np.zeros((height,breadth),dtype = np.uint8)
for j in range(height):
    for k in range(breadth):
        pixel = mat[j,k]
        if(((((pixel[0] < 26 and pixel[0] > 12) or (pixel[0] <= 115 and pixel[0] > 110)) and
             ((pixel[1] < 100 and pixel[1] > 26) or (pixel[1] < 190 and pixel[1] > 170))) or
                (pixel[0] == 0 and pixel[1] == 0 and pixel[2] < 20))) :
            img[j,k] = 255


'''Function to convert the image into b&w'''
def convert_to_bw(new_img):
    t = np.zeros(new_img.shape,np.uint8)
    for i in range(height):
        for j in range(breadth):
            if(new_img[i][j] <= 127):
                t[i][j] = 0
            else:
                t[i][j] = 255
    return t



'''Function to add padding of one layer around the image following the nearest pixel value'''
def add_padding(img):
    height,breadth = img.shape
    new_img = np.zeros((height + 4,breadth + 4))
    h,b = new_img.shape
    for i in range(height):
        for j in range(breadth):
            new_img[i+2][j+2] = img[i][j]
    #print(new_img.shape)
    new_img[0][0] = new_img[0][1] = new_img[1][0] = new_img[1][1] = img[0][0]
    new_img[0][b-1] = new_img[0][b-2] = new_img[1][b-1] = new_img[1][b-2]= img[0][breadth-1]
    new_img[h-1][b-1] = new_img[h-1][b-2] = new_img[h-2][b-1] = new_img[h-2][b-2] = img[height-1][breadth-1]
    # new_img[0][b-1] = img[0][breadth-1]
    new_img[h-1][0] = new_img[h-1][1] = new_img[h-2][0] = new_img[h-2][1] = img[height-1][0]
    for i in range(2,h-2):
        new_img[i][0] = new_img[i][1] = img[i-2][0]
        new_img[i][b-1] = new_img[i][b-2] = img[i-2][breadth-1]
    for j in range(2,b-2):
        new_img[0][j] = new_img[1][j] = img[0][j-2]
        new_img[h-1][j] = new_img[h-2][j] = img[height-1][j-2]
    return new_img


def erosion(temp_img,kernel,center,itr = 1):
    height,breadth = temp_img.shape
    eroded_image = np.zeros((height,breadth))
    #print(eroded_image)
    k_height,k_breadth = kernel.shape
    for k in range(itr):
        for i in range(2,height-4):
            for j in range(2,breadth-4):
                flag = True
                for k in range(k_height):
                    for l in range(k_breadth):
                        if(temp_img[i+k][j+l] != 255 and kernel[k][l] == 255):
                            flag = False
                            break
                    if(not(flag)):
                        break
                if(flag): 
                    eroded_image[i+center[0]][j+center[1]] = 255
        temp_img = np.copy(eroded_image)
                    
    return eroded_image



def dilation(temp_img,kernel,center,itr = 1):
    height,breadth = temp_img.shape
    dilated_image = np.zeros((height,breadth))
    #print(eroded_image)
    k_height,k_breadth = kernel.shape
    for k in range(itr):
        for i in range(2,height-4):
            for j in range(2,breadth-4):
                flag = False
                for k in range(k_height):
                    for l in range(k_breadth):
                        if(temp_img[i+k][j+l] == 255 and kernel[k][l] == 255):
                            flag = True
                            dilated_image[i+center[0]][j+center[1]] = 255
                            break
                    if(flag): 
                        break
        temp_img = np.copy(dilated_image)
                    
    return dilated_image




# def erosion(temp_img,kernel,iterations):
#     return cv2.erode(temp_img,kernel,iterations)


# def dilation(temp_img,kernel,iterations):
#     return cv2.dilate(temp_img,kernel,iterations)



def find_longest_component(image):
    max_size = 0
    max_index = None
    visited = set()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 255 and (i, j) not in visited:
                size = 0
                queue = [(i, j)]
                visited.add((i, j))
                while queue:
                    i, j = queue.pop(0)
                    size += 1

                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and \
                                image[ni][nj] == 255 and (ni, nj) not in visited:
                            queue.append((ni, nj))
                            visited.add((ni, nj))
                if size > max_size:
                    max_size = size
                    max_index = (i, j)
    queue = [max_index]
    visited = set()
    visited.add(max_index)
    pixels = set()
    while queue:
        i, j = queue.pop(0)
        pixels.add((i, j))

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and \
                    image[ni][nj] == 255 and (ni, nj) not in visited:
                queue.append((ni, nj))
                visited.add((ni, nj))

    return list(pixels)


def smoothing(img, kernel):
    k_size = kernel.shape[0]
    i_rows, i_cols = img.shape
    out_img = np.zeros_like(img)

    for i in range(k_size//2, i_rows-k_size//2):
        for j in range(k_size//2, i_cols-k_size//2):
            neighborhood = img[i-k_size//2:i+k_size//2+1, j-k_size//2:j+k_size//2+1]
            out_img[i,j] = np.sum(neighborhood * kernel, axis=(0,1))
            
    return out_img.astype(np.uint8)

def logical_and(bin_img,col_img):
    h , b , c = col_img.shape
    output = np.zeros_like(col_img)
    
    for i in range(h):
        for j in range(b):
            if(bin_img[i,j] > 0):
                output[i,j,:] = col_img[i,j,:]
            else :
                output[i,j][0] = output[i,j][1] = output[i,j][2] = 255  
    return output


kernel2 = np.ones((3,3), np.float32) / 9
kernel = np.array(((0,0,0,0,0),(0,0,255,0,0),(0,255,255,255,0),(0,0,255,0,0),(0,0,0,0,0)),np.uint8)
img = convert_to_bw(img)
img = add_padding(img)
img = dilation(img,kernel,(2,2))
img = convert_to_bw(img)
cv2.namedWindow("dilated_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("dilated_image",img)


var = find_longest_component(img)
new_img = np.zeros((height,breadth),np.uint8)
for i in var:
    new_img[i[0]][i[1]] = 255
#cv2.fillPoly(new_img, pts =[var], color=(255,255,255))

new_img = convert_to_bw(new_img)
for i in range(height):
    for j in range(breadth):
        if(new_img[i][j] <= 127):
            new_img[i][j] = 0
        else:
            new_img[i][j] = 255
cv2.namedWindow("longest_connected_component", cv2.WINDOW_AUTOSIZE)
cv2.imshow("longest_connected_component",new_img)


new_img = convert_to_bw(new_img)

new_img = dilation(new_img,kernel,(2,2))
# kernel = np.array(((255,255,255,255,255),(255,255,255,255,255),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),np.uint8)
# new_img = dilation(new_img,kernel,1)
kernel = np.array(((255,255,0,0,0),(255,255,0,0,0),(255,255,0,0,0),(255,255,0,0,0),(255,255,0,0,0)),np.uint8)
new_img = dilation(new_img,kernel,(2,2))
kernel = np.array(((0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255)),np.uint8)
new_img = dilation(new_img,kernel,(2,2))
kernel = np.array(((0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0),(255,255,255,255,255),(255,255,255,255,255)),np.uint8)
new_img = dilation(new_img,kernel,(2,2))
new_img = convert_to_bw(new_img)



kernel = np.array(((255,0,0,0,255),(255,0,0,0,255),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),np.uint8)
new_img = erosion(new_img,kernel,(2,2))
kernel = np.array(((0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255),(0,0,0,255,255)),np.uint8)
new_img = erosion(new_img,kernel,(2,2))
kernel = np.array(((0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0),(255,255,255,255,255),(255,255,255,255,255)),np.uint8)
new_img = erosion(new_img,kernel,(2,2))
kernel = np.array(((255,255,255,255,255),(255,255,255,255,255),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),np.uint8)
new_img = erosion(new_img,kernel,(2,2))
kernel = np.array(((255,0,0,0,255),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),np.uint8)
new_img = erosion(new_img,kernel,(2,2))

new_img = convert_to_bw(new_img)

new_img = smoothing(new_img,kernel2)
new_img = convert_to_bw(new_img)


final_img = logical_and(new_img,bgr_img)

cv2.namedWindow("new_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("new_image",new_img)
cv2.imwrite("Elephant.jpg",final_img)
cv2.namedWindow("Final_img",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Final_img",final_img)



cv2.waitKey(0)
cv2.destroyAllWindows()

