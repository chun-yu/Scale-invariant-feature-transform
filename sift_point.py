from PIL import Image
import numpy as np
import numba
import sys
import math
import scipy.ndimage
from scipy import signal
from PIL import ImageDraw
import cv2
#@numba.njit(fastmath=True,nogil=True)
def gaussian_filter(SD_sigma_k):
    r=6*round(SD_sigma_k)-1
    filter1=np.zeros((r,r),dtype=np.float32)
    for m in range(-int(r/2),int(r/2+1)):
        for n in range(-int(r/2),int(r/2+1)):
            filter1[m+int(r/2)][n+int(r/2)]=math.exp(-((m-r/2)**2+(n-r/2)**2)/(2*(SD_sigma_k**2)))/(2*math.pi*(SD_sigma_k**2))
    filter1=filter1/np.sum(filter1)
    return filter1

#@numba.njit(fastmath=True,nogil=True)
def gaussian_kernel_1d(SD_sigma_k):
    r=6*round(SD_sigma_k)-1
    filter1=np.zeros(r,dtype=np.float32)
    for m in range(-int(r/2),int(r/2+1)):
            filter1[m+int(r/2)]=math.exp(-(m**2)/(2*(SD_sigma_k**2)))/(math.sqrt(2*math.pi)*SD_sigma_k)
    filter1=filter1/np.sum(filter1)
    return filter1

#@numba.njit(fastmath=True,nogil=True)
def gaussian_cov1d(pixels,filter1):
    '''
    Gaussian Blur
    '''
    pixels1=np.copy(pixels)
    masksize=filter1.shape[0]
    height=pixels.shape[0]
    width=pixels.shape[1]
    mask=int(masksize/2)
    for y in range(height):
        for x in range(width):
            gray_temp=0.0
            if y>=mask and x>=mask and y<height-mask and x<width-mask: 
                pixel_temp=pixels[y][x-mask:x+mask+1]
                gray_temp=np.dot(pixel_temp, filter1)
                #gray_temp+=pixels[y-mask+m][x-mask+n]*filter1[m]
                pixels1[y][x]=gray_temp 
            '''
            else:
                
                count=0
                for m in range(masksize):
                    for n in range(masksize):
                        if ((y-mask+m>=0) and
                            (x-mask+n>=0) and
                            (y-mask+m<height) and
                            (x-mask+n<width)):
                            gray_temp+=pixels[y-mask+m][x-mask+n]*filter1[m][n]
                            count+=filter1[m][n]
                gray_temp=gray_temp/count
                pixels1[y][x]=gray_temp
            '''
    return pixels1
@numba.njit(fastmath=True,nogil=True)
def gaussian_cov(pixels,filter1):
    '''
    Gaussian Blur
    '''
    pixels1=np.copy(pixels)
    masksize=filter1.shape[0]
    height=pixels.shape[0]
    width=pixels.shape[1]
    mask=int(masksize/2)
    for y in range(height):
        for x in range(width):
            gray_temp=0.0
            if y>=mask and x>=mask and y<height-mask and x<width-mask: 
                for m in range(masksize):
                    for n in range(masksize):
                        gray_temp+=pixels[y-mask+m][x-mask+n]*filter1[m][n]
                pixels1[y][x]=gray_temp 
            '''
            else:
                count=0
                for m in range(masksize):
                    for n in range(masksize):
                        if ((y-mask+m>=0) and
                            (x-mask+n>=0) and
                            (y-mask+m<height) and
                            (x-mask+n<width)):
                            gray_temp+=pixels[y-mask+m][x-mask+n]*filter1[m][n]
                            count+=filter1[m][n]
                gray_temp=gray_temp/count
                pixels1[y][x]=gray_temp
            '''
    return pixels1

def gaussian_blur(pixels,SD_sigma,num_intervals,num_octave,assumed_blur=0.5):
    #num_images_per_octave=num_intervals+3
    k = 3 ** (1. / num_intervals)
    sig=[1.2,1.6,2.2627418,3.2,4.5254836]
    gsArray = [0,1,2,3,4]
    for i in range(5):
        #gsArray[i] = gaussian_cov(pixels,gaussian_filter(sig[i]))
        #sig[i]=sig[i]*2**(num_octave-1)
        gsArray[i]=scipy.ndimage.filters.gaussian_filter(pixels,sig[i])
    return gsArray

def DoG(gsArray):
    dfArray= [0,1,2,3]
    for i in range(1,5):
        dfArray[i-1]=(np.subtract(gsArray[i-1],gsArray[i]))
    return dfArray

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

@numba.njit(fastmath=True,nogil=True)
def findExtrema(pixels,up_pixels,sub_pixels,contrast_threshold=0.04,num_intervals=2):
    pixels1=np.zeros_like(pixels)
    height=pixels.shape[0]
    width=pixels.shape[1]
    localArea=np.zeros((3,3,3),dtype=np.float64)
    #threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    threshold=-99999
    for y in range(1,height-1):
        for x in range(1,width-1):
            if pixels[y][x]>threshold:
                currentPixel=pixels[y][x]
                localArea[0] = pixels[y-1:y+2,x-1:x+2]
                localArea[1] = up_pixels[y-1:y+2,x-1:x+2]
                localArea[2] = sub_pixels[y-1:y+2,x-1:x+2]

                maxLocal=localArea.max()
                minLocal=localArea.min()

                if(currentPixel == maxLocal) or (currentPixel == minLocal):
                    pixels1[y][x]=255

    return pixels1

def removeContrastExtrema(dfArray,pixels,sigma,octave,num_intervals=2):
    #pixels1=np.copy(pixels)
    height=pixels.shape[0]
    width=pixels.shape[1]
    for y in range(1,height-1):
        for x in range(1,width-1):
            X_update=x
            Y_update=y
            SD_update=sigma
            X_vector=0
            D_vector=0
            if pixels[y][x]==255:
                for k in range(5):
                    ##計算一階偏導數
                    dx=(dfArray[SD_update+1][Y_update][X_update+1]-dfArray[SD_update+1][Y_update][X_update-1])/2
                    dy=(dfArray[SD_update+1][Y_update][X_update+1]-dfArray[SD_update+1][Y_update][X_update-1])/2
                    ds=(dfArray[SD_update+2][Y_update][X_update+1]-dfArray[SD_update][Y_update][X_update-1])/2   
                    
                    #計算Hessian矩陣，二階偏導數
                    dxx=dfArray[SD_update+1][Y_update][X_update+1]+dfArray[SD_update+1][Y_update][X_update-1]-2*dfArray[SD_update+1][Y_update][X_update]
                    dyy=dfArray[SD_update+1][Y_update+1][X_update]+dfArray[SD_update+1][Y_update-1][X_update]-2*dfArray[SD_update+1][Y_update][X_update]
                    dss=dfArray[SD_update+2][Y_update][X_update]+dfArray[SD_update][Y_update][X_update]-2*dfArray[SD_update+1][Y_update][X_update]

                    dxy=(dfArray[SD_update+1][Y_update+1][X_update+1]-dfArray[SD_update+1][Y_update-1][X_update+1]-dfArray[SD_update+1][Y_update+1][X_update-1]+dfArray[SD_update+1][Y_update-1][X_update-1])/4
                    dys=(dfArray[SD_update+2][Y_update+1][X_update]-dfArray[SD_update+2][Y_update-1][X_update]-dfArray[SD_update][Y_update+1][X_update]+dfArray[SD_update][Y_update-1][X_update])/4
                    dsx=(dfArray[SD_update+2][Y_update][X_update+1]-dfArray[SD_update+2][Y_update][X_update-1]-dfArray[SD_update][Y_update][X_update+1]+dfArray[SD_update][Y_update][X_update-1])/4
                    Hessian_martix=np.array([[dxx,dxy,dsx],[dxy,dyy,dys],[dsx,dys,dss]],dtype=np.float64)
                    try:
                        inverse_Hessian_martix=np.linalg.inv(Hessian_martix)
                    except:
                        pixels[Y_update][X_update]=0
                        break
                    D_vector=np.array([[dx],[dy],[ds]],dtype=np.float64)
                    inverse_Hessian_martix=inverse_Hessian_martix*-1
                    X_vector=inverse_Hessian_martix.dot(D_vector) 
                     
                    if(abs(X_vector.max())>=0.5):
                        pixels[Y_update][X_update]=0
                        X_update=x+int(np.round(D_vector[0]))
                        Y_update=y+int(np.round(D_vector[1]))
                        SD_update=sigma+int(np.round(D_vector[2])) 
                        if not(X_update>=0 and X_update<width-1 and Y_update>=0 and Y_update<height-1 and SD_update>0 and SD_update<num_intervals):
                            pixels[Y_update][X_update]=0
                            break
                        if k==4:
                            pixels[Y_update][X_update]=0
                            break
                        else:
                            pixels[Y_update][X_update]=255
                                              
                
                if pixels[Y_update][X_update]==255:
                    X_offset=dfArray[SD_update+1][Y_update][X_update]+0.5*D_vector.T.dot(X_vector)
                    if(abs(X_offset)<(0.03*255)):
                        pixels[Y_update][X_update]=0
    return pixels

def removeEdgeExtrema(dfArray,pixels,i,r):
    pixels1=np.copy(pixels)
    height=pixels.shape[0]
    width=pixels.shape[1]
    for y in range(1,height-1):
        for x in range(1,width-1):
            if pixels[y][x]==255:         
                #計算Hessian矩陣，算主曲率
                dxx=dfArray[i+1][y][x+1]+dfArray[i+1][y][x-1]-2*dfArray[i+1][y][x]
                dyy=dfArray[i+1][y+1][x]+dfArray[i+1][y-1][x]-2*dfArray[i+1][y][x]
                dxy=(dfArray[i+1][y+1][x+1]-dfArray[i+1][y-1][x+1]-dfArray[i+1][y+1][x-1]+dfArray[i+1][y-1][x-1])/4
                Hessian_martix=np.array([[dxx,dxy],[dxy,dyy]],dtype=np.float64)
                D_trace=np.trace(Hessian_martix)
                D_det=np.linalg.det(Hessian_martix)
                temp=D_trace**2/D_det
                r=r
                if(temp>(r+1)**2/r):
                    pixels1[y][x]=0
    return pixels1

@numba.njit(fastmath=True,nogil=True)
def addKeypoint(keypoint,pixels,octave_index):
    height=pixels.shape[0]
    width=pixels.shape[1]    
    for y in range(height):
        for x in range(width):
            if pixels[y][x]==255:
                x1=x*(2**octave_index)
                y1=y*(2**octave_index)
                keypoint[y1][x1]=255
    return keypoint

def computeKeypoint(pixels,im,octave=3,interval=2):
    height=pixels.shape[0]
    width=pixels.shape[1]
    dog_pyramid=np.zeros((octave,interval)).tolist()
    gaussian_pyramid=np.zeros(octave).tolist()
    gaussian_pyramid[0]=pixels
    keypoint=np.zeros_like(pixels)
    for i in range(1,octave):
        im_gray=Image.fromarray(pixels.astype(np.uint8))
        width=int(width/2)
        height=int(height/2)
        im_gray=im_gray.resize((width,height),Image.NEAREST)
        im_pixels=list(im_gray.getdata())
        im_pixels = [im_pixels[i * width:(i + 1) * width] for i in range(height)]
        gaussian_pyramid[i] = np.array(im_pixels,dtype=np.float64)
    for k in range(octave):
        gsArray=gaussian_blur(gaussian_pyramid[k],1.6,2,k)
        dfArray=DoG(gsArray)
        for i in range(interval):
            dog_pyramid[k][i]=findExtrema(dfArray[i+1],dfArray[i+2],dfArray[i])
            dog_pyramid[k][i]=removeContrastExtrema(dfArray,dog_pyramid[k][i],i,k)
            dog_pyramid[k][i]=removeEdgeExtrema(dfArray,dog_pyramid[k][i],i,10)
            keypoint=addKeypoint(keypoint,dog_pyramid[k][i],k)
            
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        #im = draw_orientation(im,dog_pyramid[k],k,pixels)
        im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    return keypoint,im
def draw_orientation(im,pixels,index,ori_pixels):
    '''
    畫方向
    '''
    o_height=ori_pixels.shape[0]
    o_width=ori_pixels.shape[1]
    for two in range(2):
        height=pixels[two].shape[0]
        width=pixels[two].shape[1]
        for y in range(height):
            for x in range(width):
                if pixels[two][y][x]==255:
                    indexRow = (y)*2**index
                    indexColumn = (x)*2**index
                    all_theta=np.zeros(36).tolist()
                    maxx=int(4.5*2/(index+1))
                    for yy in range(indexRow-maxx,indexRow+maxx):
                        for xx in range(indexColumn-maxx,indexColumn+maxx):
                            if yy < 1 or xx < 1 or math.sqrt((yy-y)**2+(xx-x)**2)>maxx or yy > o_height-2 or xx > o_width-2:
                                m = 0
                            else:
                                m = math.sqrt((ori_pixels[yy+1][xx]-ori_pixels[yy-1][xx])**2+(ori_pixels[yy][xx+1]-ori_pixels[yy][xx-1])**2)
                                theta = math.atan2((ori_pixels[yy][xx+1]-ori_pixels[yy][xx-1]),(ori_pixels[yy+1][xx]-ori_pixels[yy-1][xx]))
                                all_theta[int(theta*180/math.pi/10)] = all_theta[int(theta*180/math.pi/10)] + m
                    xy_max_m = max(all_theta)
                    xy_theta = all_theta.index(xy_max_m) * 10
                    #print(xy_theta)
                    r = int(xy_max_m/255)
                    end_y = int(math.cos(math.radians(xy_theta)) * math.sqrt(r)*10 + indexRow)
                    end_x = int(math.sin(math.radians(xy_theta)) * math.sqrt(r)*10 + indexColumn)
                    #print(r)
                    if r ==0:
                        end_y = int(math.cos(math.radians(xy_theta)) * 2*5 + indexRow)
                        end_x = int(math.sin(math.radians(xy_theta)) * 2*5 + indexColumn)
                        im = cv2.arrowedLine(im, (indexColumn,indexRow), (end_x,end_y),(0,255, 0), 1,tipLength = 0.2)
                    else:
                        im = cv2.arrowedLine(im, (indexColumn,indexRow), (end_x,end_y),(0,255, 0 ), 1,tipLength = 0.2)
    return im
def drawCircle(draw,pixels):
    '''
    畫圓
    '''
    height=pixels.shape[0]
    width=pixels.shape[1]
    for y in range(height):
        for x in range(width):
            if pixels[y][x]==255:
                color=["blue"]
                r=2
                leftUpPoint = (x-r, y-r)
                rightDownPoint = (x+r, y+r)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.ellipse(twoPointList, outline=(color[0]),width=0.5,fill="blue")
if __name__ == "__main__":
    try:
        filename=sys.argv[1]            
    except:
        print("No file input and loading default file")
        filename="q2.jpg"
    im=Image.open(filename)
    im_gray = im.convert('L')
    #im_gray = im_gray.resize((300, 300))
    im_pixels=list(im_gray.getdata())
    width, height = im_gray.size
    im_pixels = [im_pixels[i * width:(i + 1) * width] for i in range(height)]
    im_pixels = np.array(im_pixels,dtype=np.float64)
    pixels,im=computeKeypoint(im_pixels,im)
    draw = ImageDraw.Draw(im)
    drawCircle(draw,pixels)
    im.show()
    im.save("result.jpg")
    #im01=Image.fromarray(im_pixels1.astype(np.uint8))
    #im01.show() 