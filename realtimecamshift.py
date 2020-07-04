import numpy as np
import cv2 as cv
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time as t
from pycuda.autoinit import context
import os
from collections import Counter
from numba import jit


mod = SourceModule("""
 #include <stdio.h> 
__global__ void Census(float *censusL,float *censusR,float *imL,float *imR,int H,int W,int N,int maxdis)
{


int d=threadIdx.x;
int h=blockIdx.x;
int w=blockIdx.y;


float count=0.0;


if((w-d)>=4 && h>=3 && h<H-3 && w<W-4)
{
for(int x=h-3;x<=h+3;x++)
{
for(int y=w-4;y<=w+4;y++)
{
if((imL[x*W+y] < imL[h*W+w]) != (imR[x*W+y-d] < imR[h*W+w-d]))
{
count++;
}
}
}
censusL[h*W*maxdis+w*maxdis+d]=count;
censusR[h*W*maxdis+(w-d)*maxdis+d]=count;
}
else
{
censusL[h*W*maxdis+w*maxdis+d]=64.0;
}

if((w+d)>=W-4 || h<3 || h>=H-3 || w<4)
{
censusR[h*W*maxdis+w*maxdis+d]=64.0;
}

}




__global__ void AD(float *imL,float *imR,float *outL,float *outR,int W,int N,int maxdis)
{


int d=threadIdx.x;
int h=blockIdx.x;
int w=blockIdx.y;
if(w-d>=0)
{
outL[h*W*maxdis+w*maxdis+d]=(fabs(imL[h*W*3+w*3]-imR[h*W*3+(w-d)*3])+fabs(imL[h*W*3+w*3+1]-imR[h*W*3+(w-d)*3+1])+fabs(imL[h*W*3+w*3+2]-imR[h*W*3+(w-d)*3+2]))/3.0;
outR[h*W*maxdis+(w-d)*maxdis+d]=outL[h*W*maxdis+w*maxdis+d];
}
else
{
outL[h*W*maxdis+w*maxdis+d]=255.0;
}
if(w+d>=W)
{
outR[h*W*maxdis+w*maxdis+d]=255.0;
}

}




__global__ void ComputCost(float *ce,float *ad,float *out,int W,int N,int maxdis)
{

int d=threadIdx.x;
int h=blockIdx.x;
int w=blockIdx.y;
int idx=h*W*maxdis+w*maxdis+d;
float lameta1=30.0;
float lameta2=10.0;
out[idx]=2.0-exp(-1*ce[idx]/lameta1)-exp(-1*ad[idx]/lameta2);

}


__global__ void get_arm(float *data,int *result,int T1,int T2,int L1,int L2,int W,int H,int N)
{
int idx=threadIdx.x + blockIdx.x * blockDim.x;
float Colordiff,Spidiff,Neidiff;
if(idx<N)
{
int h=idx/W;
int w=idx%W;
for(int l=w-1;l>=0;l--)
{
Colordiff=max(abs(data[h*W*3+w*3]-data[h*W*3+l*3]),abs(data[h*W*3+w*3+1]-data[h*W*3+l*3+1]));
Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[h*W*3+l*3+2]));
Spidiff=w-l;
Neidiff=max(abs(data[h*W*3+(l+1)*3]-data[h*W*3+l*3]),abs(data[h*W*3+(l+1)*3+1]-data[h*W*3+l*3+1]));
Neidiff=max(Neidiff,abs(data[h*W*3+(l+1)*3+2]-data[h*W*3+l*3+2]));
if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && l!=0)
continue;
if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && l!=0)
continue;
result[h*W*4+w*4]=max(int(Spidiff-1),1);
break;
}

for(int r=w+1;r<W;r++)
{
Colordiff=max(abs(data[h*W*3+w*3]-data[h*W*3+r*3]),abs(data[h*W*3+w*3+1]-data[h*W*3+r*3+1]));
Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[h*W*3+r*3+2]));
Spidiff=r-w;
Neidiff=max(abs(data[h*W*3+(r-1)*3]-data[h*W*3+r*3]),abs(data[h*W*3+(r-1)*3+1]-data[h*W*3+r*3+1]));
Neidiff=max(Neidiff,abs(data[h*W*3+(r-1)*3+2]-data[h*W*3+r*3+2]));
if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && r!=W-1)
continue;
if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && r!=W-1)
continue;
result[h*W*4+w*4+1]=max(int(Spidiff),2);
break;
}


for(int t=h-1;t>=0;t--)
{
Colordiff=max(abs(data[h*W*3+w*3]-data[t*W*3+w*3]),abs(data[h*W*3+w*3+1]-data[t*W*3+w*3+1]));
Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[t*W*3+w*3+2]));
Spidiff=h-t;
Neidiff=max(abs(data[t*W*3+w*3]-data[(t+1)*W*3+w*3]),abs(data[t*W*3+w*3+1]-data[(t+1)*W*3+w*3+1]));
Neidiff=max(Neidiff,abs(data[t*W*3+w*3+2]-data[(t+1)*W*3+w*3+2]));
if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && t!=0)
continue;
if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && t!=0)
continue;
result[h*W*4+w*4+2]=max(int(Spidiff-1),1);
break;
}

for(int b=h+1;b<H;b++)
{
Colordiff=max(abs(data[h*W*3+w*3]-data[b*W*3+w*3]),abs(data[h*W*3+w*3+1]-data[b*W*3+w*3+1]));
Colordiff=max(Colordiff,abs(data[h*W*3+w*3+2]-data[b*W*3+w*3+2]));
Spidiff=b-h;
Neidiff=max(abs(data[b*W*3+w*3]-data[(b-1)*W*3+w*3]),abs(data[b*W*3+w*3+1]-data[(b-1)*W*3+w*3+1]));
Neidiff=max(Neidiff,abs(data[b*W*3+w*3+2]-data[(b-1)*W*3+w*3+2]));
if(Spidiff<=L2 && Neidiff<T1 && Colordiff <T1 && b!=H-1)
continue;
if(L2<Spidiff && Spidiff<L1 && Neidiff<T1 && Colordiff <T2 && b!=H-1)
continue;
result[h*W*4+w*4+3]=max(int(Spidiff),2);
break;
}


}
  
}



__global__ void AggH(float *costvolume,float *out,int *result,int W,int N,int maxdis)
{

int d=threadIdx.x;
int h=blockIdx.x;
int w=blockIdx.y;
int idx=h*W*maxdis+w*maxdis+d;
out[idx]=0.0;
for(int l=w-result[h*W*4+w*4];l<w+result[h*W*4+w*4+1];l++)
{
out[idx]=out[idx]+costvolume[h*W*maxdis+l*maxdis+d];
}


}


__global__ void AggV(float *costvolume,float *out,int *result,int W,int N,int maxdis)
{

int d=threadIdx.x;
int h=blockIdx.x;
int w=blockIdx.y;
int idx=h*W*maxdis+w*maxdis+d;
out[idx]=0.0;
for(int l=h-result[h*W*4+w*4+2];l<h+result[h*W*4+w*4+3];l++)
{
out[idx]=out[idx]+costvolume[l*W*maxdis+w*maxdis+d];
}


}




__global__ void LRCheck(int *left_dis,int *right_dis,int *out,int W,int N,int maxdis)
{

int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx<N)
{
int h=idx/W;
int w=idx%W;
int d=left_dis[h*W+w];
int flag=0;
if(w-d>=0)
{
if (abs(left_dis[h*W+w]-right_dis[h*W+w-d])>1)
{
for(int x=0;x<maxdis;x++)
{
if(w-x>=0)
{
if(right_dis[h*W+w-x] > left_dis[h*W+w])
flag=1;
break;
}
}
if(flag==1)
{
out[h*W+w]=125;
}
else
{
out[h*W+w]=0;
}

}
else
{
out[h*W+w]=255;
}
}
else
{
out[idx]=0;
}
}
}


__global__ void Postprocess(int *ref,int *disimg,int W,int H,int N)
{
int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx<N)
{
int h=idx/W;
int w=idx%W;
int ah=h;
int aw=w;
int index=0;
int val[8];
int temp;
int symbol=0;
if(ref[h*W+w]==125 || ref[h*W+w]==0)
{
for(int x=-1;x<=1;x++)
{
for(int y=-1;y<=1;y++)
{
if(x==0&&y==0)
continue;
while(ah+x>=0 && ah+x<H && aw+y>=0 && aw+y<W && ref[(ah+x)*W+(aw+y)] !=255)
{
ah=ah+x;
aw=aw+y;
}
ah=ah+x;
aw=aw+y;
if(ah>0 && ah<H && aw>0 && aw<W)
{
val[index]=disimg[ah*W+aw];
++index;
}
ah=h;
aw=w;
}
}

for (int i=0; i<index-1; ++i) 
{
for (int j=0; j<index-1-i; ++j) 
{
 if (val[j] < val[j+1])
{
  temp = val[j];
  val[j] = val[j+1];
  val[j+1] = temp;
}
}
}
}

if(ref[h*W+w]==0)
{
for(int x=-1;x<=1;x++)
{
for(int y=-1;y<=1;y++)
{
if(x==0&&y==0)
continue;
if(ah+x>=0 && ah+x<H && aw+y>=0 && aw+y<W )
{
ah=ah+x;
aw=aw+y;
if(ref[ah*W+aw]==125)
{
symbol=1;
}
ah=h;
aw=w;
}
}
}
if(symbol==1)
{
disimg[h*W+w]=val[index-2];
}
else
{
if(index%2==1)
disimg[h*W+w]=val[int(index/2)];
else
disimg[h*W+w]=int((val[int(index/2)-1]+val[int(index/2)])/2);
}
}

if(ref[h*W+w]==125)
{
disimg[h*W+w]=val[index-2];
}

 
}
}


__global__ void Iterative_Region_Voting(int *disimg,int *result,int *ref,int *refnew,int W,int N,int maxdis)
{
int idx=threadIdx.x + blockIdx.x*blockDim.x;
if(idx<N)
{
int h=idx/W;
int w=idx%W;
int val[256];
for(int j=0;j<256;j++)
{
 val[j]=0;
}
int count=0;
int newval=0;
int temp=0;
int Ts=20;
float Th=0.4;
refnew[h*W+w]=ref[h*W+w];

if(ref[h*W+w] != 255)
{
for(int t=h-result[h*W*4+w*4+2];t<h+result[h*W*4+w*4+3];t++)
{
for(int l=w-result[t*W*4+w*4];l<w+result[t*W*4+w*4+1];l++)
{
if(ref[t*W+l]==255)
{
val[disimg[t*W+l]]++;
count++;
}
}
}


for(int i=0;i<maxdis;i++)
{
if(val[i]>temp)
{
newval=i;
temp=val[i];
} 
}
__syncthreads();


if(count > Ts)
{
if(float(temp/count)>Th)
{
refnew[h*W+w]=255;
disimg[h*W+w]=newval;
}
}
}
} 
}

__global__ void select_disparity(float *cost,int *out,int W,int N,int maxdis)
{
int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx<N)
{
int h=idx/W;
int w=idx%W;
int index=0;
int min=cost[h*W*maxdis+w*maxdis+index];
for(int d=0;d<maxdis;d++)
{
if(cost[h*W*maxdis+w*maxdis+d]<min)
{
index=d;
min=cost[h*W*maxdis+w*maxdis+d];
} 
}
out[h*W+w]=index;

}
}






""")


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        
        
def to3d(Q,x,y,disimg):
    trige=np.array([x,y,disimg[x,y],1],dtype=np.float32).reshape(4,1)
    temp=np.dot(Q,trige)
    return temp/temp[3][0]
    
@jit
def subfuction(h,w,result,disimg,lrc,Ts,Th,maxdis):
    count=0;val=np.zeros(shape=maxdis,dtype=np.uint8)
    for y in range(h-result[h,w,2],h+result[h,w,3]):
        for x in range(w-result[y,w,0],w+result[y,w,1]):
            if lrc[y,x]==255:
                val[disimg[y,x]]=val[disimg[y,x]]+1
                count=count+1
    if count<=Ts:
        return disimg[h,w],lrc[h,w]
    else:
        time=np.max(val)
        if time/count >Th:
            return np.argmax(val),255
        else:
            return disimg[h,w],lrc[h,w]
    
@jit
def Iterative_Region_Voting(disimg,lrc,result,Ts,Th,maxdis):
    lrca=lrc.copy()
    for h in range(lrc.shape[0]):
        for w in range(lrc.shape[1]):
            if lrc[h,w]!=255:
                disimg[h,w],lrca[h,w]=subfuction(h,w,result,disimg,lrc,Ts,Th,maxdis)
    return disimg,lrca

@jit                      
def Sub_pixel_Enhancement(disimg,costVolume):
    disimg=disimg.astype(np.float32)
    for h in range(0,disimg.shape[0]):
        for w in range(0,disimg.shape[1]):
            d0=int(disimg[h,w]);d1=d0-1;d2=d0+1
            if (costVolume[h,w,d2]+costVolume[h,w,d1]-2*costVolume[h,w,d0])!=0 and d1>=0 and d2<MaxDis:
                disimg[h,w]=disimg[h,w]-((costVolume[h,w,d2]-costVolume[h,w,d1])/\
                                        (2*(costVolume[h,w,d2]+costVolume[h,w,d1]-2*costVolume[h,w,d0])))
    return disimg



def normalize(volume, maxdisparity):
    return 255.0 * volume / maxdisparity


def comput_distance(b,f,d):
    return (b*f)/(d*10)

def spatial_pos(b,f,d,h,w):
    z=(b*f)/d
    x=w*z/f
    y=h*z/f
    return x,y,z


def mouse_callback(event,x,y,flags,param):
    draw_img=param.copy();
    if event==1:
        posh.append(y);posw.append(x)
        cv.line(draw_img,(posw[0],posh[0]),(posw[0],posh[0]),(0,255,0),thickness=5)
        cv.imshow("origin",draw_img)
    elif flags==1:
        preval.append(edge_output[y,x])
        try:
            if preval[-1]==255 and (preval[-2]+preval[-1]==255):
                point.append(y);point.append(x)
                #print('ok')
        except IndexError:
            pass
    elif event==4:
        posh.append(y);posw.append(x)
        if posh[0]==posh[1] and posw[0]==posw[1]:
            x0,y0,z0,_a=to3d(Q,posh[1],posw[1],disp)
            print('pos'+'('+str(posh[1])+','+str(posw[1])+')'+' 视差值'+str(disp[posh[1],posw[1]])[:4]+' distance:({:.0f})MM'.format(z0[0]))
            cv.line(draw_img,(posw[0],posh[0]),(posw[1],posh[1]),(0,255,0),thickness=2)
            temp=frame.copy()
            cv.line(temp,(posw[0],posh[0]),(posw[0]+640-int(round(disp[posh[0],posw[0]])),posh[0]),(0,0,255),thickness=1)
            print('X:'+str(round(x0[0]))+' Y:'+str(round(y0[0]))+' Z:'+str((round(z0[0]))))
            cv.putText(temp,str(order[0]),(posw[0],posh[0]),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),1) 
            order[0]=order[0]+1
            cv.imshow("origin",draw_img)
            cv.imshow("frame",temp)
            posh.clear();posw.clear()
        else:
            try:
                cv.line(imCR,(posw[0],posh[0]),(posw[1],posh[1]),(0,255,0),thickness=1)
                cv.imshow("origin",imCR)
                t1=posh.pop();t2=posw.pop()  
                x0,y0,z0,_a=to3d(Q,posh[0],posw[0],disp)
                x1,y1,z1,_a=to3d(Q,t1,t2,disp)
                length=np.sqrt(np.square(x0-x1)+np.square(y0-y1)+np.square(z0-z1))
                print('绿线:'+'{:.0f} MM'.format(round(length[0])))
                print('绿线起始点'+'('+str(posh[0])+','+str(posw[0])+')'+'处的视差值:'+'{:.1f}'.format(disp[posh[0],posw[0]]))
                print('X:'+str(round(x0[0]))+' Y:'+str(round(y0[0]))+' Z:'+str((round(z0[0]))))
                print('绿线终点'+'('+str(t1)+','+str(t2)+')'+'处的视差值:'+'{:.1f}'.format(disp[t1,t2]))
                print('X:'+str(round(x1[0]))+' Y:'+str(round(y1[0]))+' Z:'+str((round(z1[0]))))
                x0,y0,z0,_a=to3d(Q,point[0],point[1],disp)
                x1,y1,z1,_a=to3d(Q,point[-2],point[-1],disp)
                length=np.sqrt(np.square(x0-x1)+np.square(y0-y1)+np.square(z0-z1))
                print('红线:'+'({:.0f}) MM'.format(round(length[0])))
                print('红线起始点'+'('+str(point[0])+','+str(point[1])+')'+'处的视差值:'+'{:.1f}'.format(disp[point[0],point[1]]))
                print('X:'+str(round(x0[0]))+' Y:'+str(round(y0[0]))+' Z:'+str((round(z0[0]))))
                print('红线终点'+'('+str(point[-2])+','+str(point[-1])+')'+'处的视差值:'+'{:.1f}'.format(disp[point[-2],point[-1]]))
                print('X:'+str(round(x1[0]))+' Y:'+str(round(y1[0]))+' Z:'+str((round(z1[0]))))
                print('-------------------------------------------------')
                cv.putText(imCR,str(order[0]),(point[1],point[0]),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),1)
                cv.line(imCR,(point[1],point[0]),(point.pop(),point.pop()),(0,0,255),thickness=1)
                cv.imshow("origin",imCR)
                order[0]=order[0]+1
            except IndexError:
                print('-------------------------------------------------')
            posh.clear();posw.clear();point.clear();preval.clear();
        
        
    




left_camera_matrix=np.array([[425.6360,0,0],
                             [0,425.4670,0],
                             [351.1606,262.5314,1]]).T

left_distortion=np.array([0.0335,-0.0373,0.0000,0.0000,0.0000])


right_camera_matrix=np.array([[425.3834,0,0],
                              [0,425.1154,0],
                              [347.1651,256.0538,1]]).T
right_distortion=np.array([0.0293,-0.0290,0.0000,0.0000,0.0000])

mutural_R_matrix=np.array([[1.0000,-0.006,0.0019],
                           [0.0060,1.0000,-0.0021],
                           [-0.0019,0.0022,1.0000]]).T

Translation=np.array([-58.6745,-0.1516,-0.1345])
baseline=58.6745
size=(640,480)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size,
                                                                  mutural_R_matrix,Translation)

left_map1, left_map2 = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv.CV_16SC2)
cap=cv.VideoCapture(1)
cap.set(3,1280)
cap.set(4,480);
posh=[];posw=[];coinval=[]





cv.namedWindow("origin",cv.WINDOW_AUTOSIZE)
cv.namedWindow("depth",cv.WINDOW_AUTOSIZE)
cv.namedWindow("frame",cv.WINDOW_AUTOSIZE)
roi=None;tracker=cv.TrackerMedianFlow_create();originfram=None
Census=mod.get_function("Census")
AD=mod.get_function("AD")
ComputCost=mod.get_function("ComputCost")
get_arm=mod.get_function("get_arm")
AggH=mod.get_function("AggH")
AggV=mod.get_function("AggV")
LRCheck=mod.get_function("LRCheck")
Postprocess=mod.get_function("Postprocess")
select_disparity=mod.get_function("select_disparity")
#Iterative_Region_Voting=mod.get_function("Iterative_Region_Voting")
#sub_enhance=mod.get_function("sub_enhance")









H=480;W=640;MaxDis=80;N=H*W*MaxDis;NN=H*W

T1=23;T2=11;L1=40;L2=20;











point=[];preval=[]
left_census = drv.mem_alloc(H*W*MaxDis*4)
right_census = drv.mem_alloc(H*W*MaxDis*4)
left_ad = drv.mem_alloc(H*W*MaxDis*4)
right_ad = drv.mem_alloc(H*W*MaxDis*4)
left_cost = drv.mem_alloc(H*W*MaxDis*4)
right_cost = drv.mem_alloc(H*W*MaxDis*4)
resultL=drv.mem_alloc(H*W*4*4)
resultR=drv.mem_alloc(H*W*4*4)
left_dis=drv.mem_alloc(H*W*4)
right_dis=drv.mem_alloc(H*W*4)
final=drv.mem_alloc(H*W*4)
finalnew=drv.mem_alloc(H*W*4)
imCLdata=drv.mem_alloc(H*W*3*4)
imCRdata=drv.mem_alloc(H*W*3*4)
leftdis=np.zeros(shape=(H,W),dtype=np.int32)
resultLL=np.zeros(shape=(H,W,4),dtype=np.int32)
lrc=np.zeros(shape=(H,W),dtype=np.int32)
left_agged=np.zeros(shape=(H,W,MaxDis),dtype=np.int32)
while True:
    order=[1]
    ret,frame=cap.read()
    if ret!=True:
        print('error find camera')
        break
    origin_left=frame[:,0:640,:]
    origin_right=frame[:,640:1280,:] 
    cv.imshow("origin",origin_left)
    c=cv.waitKey(40)
    if c==32:
        start=t.time()
        imCL = cv.remap(origin_left, left_map1,left_map2, cv.INTER_CUBIC)
        imCR = cv.remap(origin_right, right_map1,right_map2, cv.INTER_CUBIC)
        cv.imwrite("leftt.png",imCL)
        cv.imwrite("rightt.png",imCR)
        frame[:,0:640,:]=imCL
        frame[:,640:1280,:]=imCR
        #imCL=cv.imread(r'leftt.png')
        #imCR=cv.imread(r'rightt.png')
        imL=cv.cvtColor(imCL,cv.COLOR_BGR2GRAY)
        imR=cv.cvtColor(imCR,cv.COLOR_BGR2GRAY)
        #edge_output = cv.Canny(imL, 250, 750)
        edge_output = cv.Canny(imL,120,300)
        edge_output=edge_output.astype(np.int32)            
        imCLd=imCL.reshape(H*W*3).astype(np.float32)
        imCRd=imCR.reshape(H*W*3).astype(np.float32)
        drv.memcpy_htod(imCLdata,imCLd)
        drv.memcpy_htod(imCRdata,imCRd)
        imLdata=imL.astype(np.float32)
        imRdata=imR.astype(np.float32)
        Census(left_census,right_census,drv.In(imLdata),drv.In(imRdata),np.uint32(H),np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AD(imCLdata,imCRdata,left_ad,right_ad,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        ComputCost(left_census,left_ad,left_cost,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        ComputCost(right_census,right_ad,right_cost,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        get_arm(imCLdata,resultL,np.uint32(T1),np.uint32(T2),np.uint32(L1),np.uint32(L2),np.uint32(W),np.uint32(H),np.uint32(NN),block=(1024,1,1),grid=(4800,1,1))
        get_arm(imCRdata,resultR,np.uint32(T1),np.uint32(T2),np.uint32(L1),np.uint32(L2),np.uint32(W),np.uint32(H),np.uint32(NN),block=(1024,1,1),grid=(4800,1,1))
        AggH(left_cost,left_census,resultL,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggV(left_census,left_cost,resultL,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggH(left_cost,left_census,resultL,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggV(left_census,left_cost,resultL,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggH(right_cost,right_census,resultR,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggV(right_census,right_cost,resultR,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggH(right_cost,right_census,resultR,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        AggV(right_census,right_cost,resultR,np.uint32(W),np.uint32(N),np.uint32(MaxDis),block=(MaxDis,1,1),grid=(H,W,1))
        select_disparity(left_cost,left_dis,np.uint32(W),np.uint32(NN),np.uint32(MaxDis),block=(1024,1,1),grid=(4800,1,1))
        select_disparity(right_cost,right_dis,np.uint32(W),np.uint32(NN),np.uint32(MaxDis),block=(1024,1,1),grid=(4800,1,1))
        LRCheck(left_dis,right_dis,final,np.uint32(W),np.uint32(NN),block=(1024,1,1),grid=(4800,1,1))
        context.synchronize()
        drv.memcpy_dtoh(leftdis,left_dis)
        drv.memcpy_dtoh(resultLL,resultL)
        drv.memcpy_dtoh(lrc,final)
        drv.memcpy_dtoh(left_agged,left_cost)
        for i in range(5):
            leftdis,lrc=Iterative_Region_Voting(leftdis,lrc,resultLL,20,0.4,MaxDis)
            #cv.imwrite("finadded.png",cv.applyColorMap(np.uint8(normalize(leftdis,MaxDis)), cv.COLORMAP_JET))
        Postprocess(drv.InOut(lrc),drv.InOut(leftdis),np.uint32(W),np.uint32(H),np.uint32(NN),block=(1024,1,1),grid=(4800,1,1))
        ccdd=Sub_pixel_Enhancement(leftdis,left_agged)
        #ccdd=leftdis.copy()
        ccdd=cv.medianBlur(ccdd,3)
        posted=np.round(ccdd).astype(np.uint8)
        cv.imwrite("posted.png",posted)
        cv.imshow("depth",cv.applyColorMap(np.uint8(normalize(ccdd,MaxDis)), cv.COLORMAP_JET))
        disp = ccdd.astype(np.float32)
        points = cv.reprojectImageTo3D(ccdd,Q)
        colors = cv.cvtColor(imCL, cv.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
        write_ply(out_fn, out_points, out_colors)
        #print('%s saved' % out_fn)
        end=t.time()
        #print('time cost...({:.2f}S)'.format(end-start))
        print('done...')
        imCR[:,:,0]=imCL[:,:,0]
        imCR[:,:,1]=np.bitwise_or(imCL[:,:,1],np.uint8(edge_output))
        imCR[:,:,2]=np.bitwise_or(imCL[:,:,2],np.uint8(edge_output))
        """
        if roi:
            ret, bbox = tracker.update(imCL)
            if bbox[2]>(roi[2]+50) or bbox[3]>(roi[3]+50):
                del tracker
                tracker=cv.TrackerMedianFlow_create()
                tracker.init(originfram,roi)
            if ret:
                #print('ok')
                val=leftdis[int(max(bbox[1],0)):int(bbox[1]+bbox[3]),int(max(bbox[0],0)):int(bbox[0]+bbox[2])]
                dis_val=Counter(val.ravel()).most_common(1)[0][0]
                trige=np.array([int(bbox[0]+(bbox[2]/2)),int(bbox[1]+(bbox[3]/2)),dis_val,1],dtype=np.float32).reshape(4,1)
                postion=np.dot(Q,trige)
                posz=(postion[2][0]*baseline)/dis_val
                print('X:'+str(postion[0]*(-1))+' Y:'+str(postion[1]*(-1))+' Z:'+'['+str(posz)+']')
            else:
                print('false')
            cv.rectangle(imCR,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),(0,0,255),thickness=2)
        """
        cv.imshow("origin",imCR)
        cv.setMouseCallback("origin",mouse_callback,imCR)
        key=cv.waitKey(0)
        """
        if key==32:
            roi=cv.selectROI("origin",imCL)
            originfram=imCL
            tracker=cv.TrackerMedianFlow_create()
            tracker.init(imCL,roi)
            print(roi)
        """
        if key==27:
            cv.destroyAllWindows()
            cap.release()
            cv.imwrite("frame.png",frame)
            break;
cv.destroyAllWindows()


























