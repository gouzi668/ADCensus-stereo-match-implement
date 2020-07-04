import numpy as np
import cv2 as cv
import time as tt
from numba import jit




def normalize(volume, maxdisparity):
    return 255.0 * volume / maxdisparity

def select_disparity(aggregation_volume):
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map





@jit
def Census(imL,imR,maxdis):
    H=imL.shape[0];W=imL.shape[1]
    finalR=np.zeros(shape=(H,W,maxdis),dtype=np.float32)
    finalL=np.zeros(shape=(H,W,maxdis),dtype=np.float32)
    finalL[:,0:4,:]=64.0;finalL[:,(W-4):W,:]=64.0
    finalL[0:3,:,:]=64.0;finalL[(H-3):H,:,:]=64.0
    finalR[:,:,:]=64.0;count=0.0
    for h in range(3,H-3):
        for w in range(4,W-4):
            for d in range(0,maxdis):
                if (w-d) < 4:
                    finalL[h,w,d:]=64.0
                    break
                else:
                    for x in range(h-3,h+4):
                        for y in range(w-4,w+5):
                            if (imL[x,y]<imL[h,w])!=(imR[x,y-d]<imR[h,w-d]):
                                count=count+1.0
                    finalL[h,w,d]=count
                    finalR[h,w-d,d]=count
                    count=0
    return finalL,finalR



@jit
def Computer_AD(left_color,right_color,MaxDis):
    left_color=left_color.astype(np.float32)
    right_color=right_color.astype(np.float32)
    CadL=np.zeros(shape=(left_color.shape[0],left_color.shape[1],MaxDis),dtype=np.float32)
    CadR=np.zeros(shape=(left_color.shape[0],left_color.shape[1],MaxDis),dtype=np.float32)
    CadL[:,:,:]=255.0;CadR[:,:,:]=255.0
    for h in range(left_color.shape[0]):
        for w in range(left_color.shape[1]):
            for d in range(MaxDis):
                if w-d>=0:
                    CadL[h,w,d]=np.sum(np.abs(left_color[h,w]-right_color[h,w-d]))/3.0
                    CadR[h,w-d,d]=CadL[h,w,d]
    return CadL,CadR


def CostVolume(Cc,Ce,a1,a2,H,W,MaxDis):
    CostVolume=np.zeros(shape=(H,W,MaxDis),dtype=np.float32)
    CostVolume[:,:,:]=2.0
    Cc=Cc*(-1/a1)
    Ce=Ce*(-1/a2)
    CostVolume=CostVolume-np.exp(Cc)-np.exp(Ce)
    return CostVolume

@jit
def Check(Colordiff,Spidiff,Neidiff,T1,T2,L1,L2):
    if Spidiff<=L2 and Neidiff<T1 and Colordiff <T1:
        return False
    if L2<Spidiff<L1 and Neidiff<T1 and Colordiff <T2:
        return False
    return True

@jit
def get_arm(img,T1,T2,L1,L2):
    H=img.shape[0];W=img.shape[1]
    img=img.astype(np.int32)
    result=np.zeros(shape=(H,W,4),dtype=np.uint8)
    for h in range(0,H):
        for w in range(0,W):
            for l in range(w-1,-1,-1):
                Colordiff=max(np.abs(img[h,w]-img[h,l]))
                Spidiff=w-l
                Neidiff=max(np.abs(img[h,l]-img[h,l+1]))
                if Check(Colordiff,Spidiff,Neidiff,T1,T2,L1,L2) or l==0:
                    result[h,w,0]=max(Spidiff-1,1)
                    break           
            for r in range(w+1,W):
                Colordiff=max(np.abs(img[h,w]-img[h,r]))
                Spidiff=r-w
                Neidiff=max(np.abs(img[h,r]-img[h,r-1]))
                if Check(Colordiff,Spidiff,Neidiff,T1,T2,L1,L2) or r==W-1:
                    result[h,w,1]=max(Spidiff,2)
                    break
            for t in range(h-1,-1,-1):
                Colordiff=max(np.abs(img[t,w]-img[h,w]))
                Spidiff=h-t
                Neidiff=max(np.abs(img[t,w]-img[t+1,w]))
                if Check(Colordiff,Spidiff,Neidiff,T1,T2,L1,L2) or t==0:
                    result[h,w,2]=max(Spidiff-1,1)
                    break
            for b in range(h+1,H):
                Colordiff=max(np.abs(img[h,w]-img[b,w]))
                Spidiff=b-h
                Neidiff=max(np.abs(img[b,w]-img[b-1,w]))
                if Check(Colordiff,Spidiff,Neidiff,T1,T2,L1,L2) or b==H-1:
                    result[h,w,3]=max(Spidiff,2)
                    break
    return result




@jit
def aggodd(result,CostVolume):
    agged=np.zeros(shape=(CostVolume.shape[0],CostVolume.shape[1],CostVolume.shape[2]),dtype=np.float32)
    agg=np.zeros(shape=(CostVolume.shape[0],CostVolume.shape[1],CostVolume.shape[2]),dtype=np.float32)
    for h in range(CostVolume.shape[0]):
        for w in range(CostVolume.shape[1]):
            #for d in range(CostVolume.shape[2]):
            for l in range(w-result[h,w,0],w+result[h,w,1]):
                agg[h,w,:]=agg[h,w,:]+CostVolume[h,l,:]
    for h in range(CostVolume.shape[0]):
        for w in range(CostVolume.shape[1]):
            #for d in range(CostVolume.shape[2]):
            for t in range(h-result[h,w,2],h+result[h,w,3]):
                agged[h,w,:]=agged[h,w,:]+agg[t,w,:]
    return agged



@jit
def aggeven(result,CostVolume):
    agged=np.zeros(shape=(CostVolume.shape[0],CostVolume.shape[1],CostVolume.shape[2]),dtype=np.float32)
    agg=np.zeros(shape=(CostVolume.shape[0],CostVolume.shape[1],CostVolume.shape[2]),dtype=np.float32)
    for h in range(CostVolume.shape[0]):
        for w in range(CostVolume.shape[1]):
            for d in range(CostVolume.shape[2]):
                for t in range(h-result[h,w,2],h+result[h,w,3]):
                    agged[h,w,d]=agged[h,w,d]+CostVolume[t,w,d]
    for h in range(CostVolume.shape[0]):
        for w in range(CostVolume.shape[1]):
            for d in range(CostVolume.shape[2]):
                for l in range(w-result[h,w,0],w+result[h,w,1]):
                    agg[h,w,d]=agg[h,w,d]+agged[h,l,d]
    return agg



@jit
def penalty(OP1,OP2,thres,val,val1,maxdisparity,cur_d):
    penalties = np.zeros(shape=(maxdisparity),dtype=np.float32)
    if val<thres and val1<thres:
        P1=OP1;P2=OP2
    if val>thres and val1>thres:
        P1=OP1/10.0;P2=OP1/10.0
    if (val-thres)*(val1-thres)<0:
        P1=OP1/4.0;P2=OP2/4.0
    for i in range(maxdisparity):
        if np.abs(i-cur_d)==1:
            penalties[i]=P1
        elif np.abs(i-cur_d)>1:
            penalties[i]=P2

    return penalties


@jit
def agglr(costVolume,color_left,color_right,maxDis,P1,P2,thres):
    H=costVolume.shape[0];W=costVolume.shape[1]
    imgL=color_left.astype(np.float32);imgR=color_right.astype(np.float32)
    penalties=np.zeros(shape=(maxDis),dtype=np.float32)
    aggtwo=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggfour=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggtwo[:,0,:]=costVolume[:,0,:]
    aggfour[:,W-1,:]=costVolume[:,W-1,:]
    for w in range(1,W):
        for h in range(0,H):
            val=max(np.abs(imgL[h,w]-imgL[h,w-1]))
            for d in range(maxDis):
                if w-d-1>=0:                    
                    val1=max(np.abs(imgR[h,w-d-1]-imgR[h,w-d]))
                else:
                    val1=val+1                  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggtwo[h,w,d]=costVolume[h,w,d]+np.min(aggtwo[h,w-1,:]+penalties)-np.min(aggtwo[h,w-1,:])
                
    for w in range(W-2,-1,-1):
        for h in range(0,H):
            val=max(np.abs(imgL[h,w]-imgL[h,w+1]))
            for d in range(maxDis):
                if w-d>=0:                   
                    val1=max(np.abs(imgR[h,w-d+1]-imgR[h,w-d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggfour[h,w,d]=costVolume[h,w,d]+np.min(aggfour[h,w+1,:]+penalties)-np.min(aggfour[h,w+1,:])
    return aggtwo,aggfour                            
                
@jit
def aggtb(costVolume,color_left,color_right,maxDis,P1,P2,thres):
    H=costVolume.shape[0];W=costVolume.shape[1]
    imgL=color_left.astype(np.float32);imgR=color_right.astype(np.float32)
    penalties=np.zeros(shape=(maxDis),dtype=np.float32)
    aggone=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggthree=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggone[0,:,:]=costVolume[0,:,:]    
    aggthree[H-1,:,:]=costVolume[H-1,:,:]
    for h in range(1,H):
        for w in range(0,W):
            val=max(np.abs(imgL[h-1,w]-imgL[h,w]))
            for d in range(maxDis):
                if w-d>=0:                    
                    val1=max(np.abs(imgR[h-1,w-d]-imgR[h,w-d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggone[h,w,d]=costVolume[h,w,d]+np.min(aggone[h-1,w,:]+penalties)-np.min(aggone[h-1,w,:])
                
    for h in range(H-2,-1,-1):
        for w in range(0,W):
            val=max(np.abs(imgL[h+1,w]-imgL[h,w]))
            for d in range(maxDis):
                if w-d>=0:                    
                    val1=max(np.abs(imgR[h+1,w-d]-imgR[h,w-d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggthree[h,w,d]=costVolume[h,w,d]+np.min(aggthree[h+1,w,:]+penalties)-np.min(aggthree[h+1,w,:])
    return aggone,aggthree  



@jit
def agglr1(costVolume,color_left,color_right,maxDis,P1,P2,thres):
    H=costVolume.shape[0];W=costVolume.shape[1]
    imgL=color_left.astype(np.float32);imgR=color_right.astype(np.float32)
    penalties=np.zeros(shape=(maxDis),dtype=np.float32)
    aggtwo=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggfour=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggtwo[:,0,:]=costVolume[:,0,:]
    aggfour[:,W-1,:]=costVolume[:,W-1,:]
    for w in range(1,W):
        for h in range(0,H):
            val=max(np.abs(imgR[h,w]-imgR[h,w-1]))
            for d in range(maxDis):
                if w+d<W:                    
                    val1=max(np.abs(imgL[h,w+d-1]-imgR[h,w+d]))
                else:
                    val1=val+1                  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggtwo[h,w,d]=costVolume[h,w,d]+np.min(aggtwo[h,w-1,:]+penalties)-np.min(aggtwo[h,w-1,:])
                
    for w in range(W-2,-1,-1):
        for h in range(0,H):
            val=max(np.abs(imgR[h,w]-imgR[h,w+1]))
            for d in range(maxDis):
                if w+d+1<W:                    
                    val1=max(np.abs(imgL[h,w+d+1]-imgL[h,w+d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggfour[h,w,d]=costVolume[h,w,d]+np.min(aggfour[h,w+1,:]+penalties)-np.min(aggfour[h,w+1,:])
    return aggtwo,aggfour                            
                
@jit
def aggtb1(costVolume,color_left,color_right,maxDis,P1,P2,thres):
    H=costVolume.shape[0];W=costVolume.shape[1]
    imgL=color_left.astype(np.float32);imgR=color_right.astype(np.float32)
    penalties=np.zeros(shape=(maxDis),dtype=np.float32)
    aggone=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggthree=np.zeros(shape=(H, W, maxDis), dtype=np.float32)
    aggone[0,:,:]=costVolume[0,:,:]    
    aggthree[H-1,:,:]=costVolume[H-1,:,:]
    for h in range(1,H):
        for w in range(0,W):
            val=max(np.abs(imgR[h-1,w]-imgR[h,w]))
            for d in range(maxDis):
                if w+d<W:  
                    val1=max(np.abs(imgL[h-1,w+d]-imgL[h,w+d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggone[h,w,d]=costVolume[h,w,d]+np.min(aggone[h-1,w,:]+penalties)-np.min(aggone[h-1,w,:])
                
    for h in range(H-2,-1,-1):
        for w in range(0,W):
            val=max(np.abs(imgR[h+1,w]-imgR[h,w]))
            for d in range(maxDis):
                if w-d>=0:                    
                    val1=max(np.abs(imgL[h+1,w+d]-imgL[h,w+d]))
                else:
                    val1=val+1  
                penalties=penalty(P1,P2,thres,val,val1,maxDis,d)
                aggthree[h,w,d]=costVolume[h,w,d]+np.min(aggthree[h+1,w,:]+penalties)-np.min(aggthree[h+1,w,:])
    return aggone,aggthree 




def ClassifyOutlier(left_dis,right_dis,maxdis):
    final=np.zeros(shape=(left_dis.shape[0],left_dis.shape[1]),dtype=np.uint8)
    a=0;b=0
    for h in range(left_dis.shape[0]):
        for w in range(left_dis.shape[1]):
            d=left_dis[h,w]
            if w-d >=0:
                if np.abs(left_dis[h,w]-right_dis[h,w-d])<=1:
                    final[h,w]=255
                    a=a+1
                else:
                    for x in range(0,maxdis):
                        if w-x >=0:
                            if right_dis[h,w-x]>left_dis[h,w]:
                                final[h,w]=125
                                b=b+1
                                break
    return final,a,b,H*W-a-b

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
def edge_optimize(edge,disimg,costVolume):
    for h in range(3,disimg.shape[0]-3):
        for w in range(4,disimg.shape[1]-4):
            d1=disimg[h,w];d0=disimg[h,w-1];d2=disimg[h,w+1]
            if edge[h,w]==255:
                if costVolume[h,w-1,d0]< costVolume[h,w,d1]:
                    disimg[h,w]=disimg[h,w-1]
                elif costVolume[h,w+1,d2] < costVolume[h,w,d1]:
                    disimg[h,w]=disimg[h,w+1]
    return disimg

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
                        
   
          

def occluder(posx,posy,img,ref):
    direction=[[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]
    val=[];H=img.shape[0];W=img.shape[1];a=posx;b=posy
    for dir in direction:
        while 0<=posx+dir[0]<H and 0<=posy+dir[1]<W and ref[posx+dir[0],posy+dir[1]]!=255:
            posx+=dir[0];posy+=dir[1]
        if 0<posx+dir[0]<H-1 and 0<posy+dir[1]<W-1:
            val.append(img[posx+dir[0],posy+dir[1]])
        posx=a;posy=b
    val[np.argmin(val)]=255
    return min(val)

def mismatcher(posx,posy,img,ref):
    direction=[[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]
    val=[];H=img.shape[0];W=img.shape[1];a=posx;b=posy
    for dir in direction:
        while 0<=posx+dir[0]<H and 0<=posy+dir[1]<W and ref[posx+dir[0],posy+dir[1]]!=255:
            posx+=dir[0];posy+=dir[1]
        if 0<posx+dir[0]<H-1 and 0<posy+dir[1]<W-1:
            val.append(img[posx+dir[0],posy+dir[1]])
        posx=a;posy=b
    val.sort()
    if len(val)%2==1:
        value=val[int(len(val)/2)]
    else:
        value=int((val[int(len(val)/2)-1]+val[int(len(val)/2)])/2)
    return value

def mistocc(posx,posy,ref):
    direction=[[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]
    H=ref.shape[0];W=ref.shape[1];num=0
    for dir in direction:
        if 0<=posx+dir[0]<H and 0<=posy+dir[1]<W and ref[posx+dir[0],posy+dir[1]]==125:
            num+=1
    if not num==0:
        return True
    else:
        return False


def process(ref,img):
    H=ref.shape[0];W=ref.shape[1]
    for h in range(0,H):
        for w in range(0,W):
            if ref[h,w]==0:
                if mistocc(h,w,ref):
                    img[h,w]=occluder(h,w,img,ref)
                else:
                    img[h,w]=mismatcher(h,w,img,ref)
            elif ref[h,w]==125:
                    img[h,w]=occluder(h,w,img,ref)

    return img


           


                

start=tt.time()
print('start...')
reff=cv.imread(r'disp2.png',0)
ref=reff.astype(np.float32)
ref=ref/4.0
left_color=cv.imread(r'aa.png')
right_color=cv.imread(r'bb.png')
left_gray=cv.cvtColor(left_color,cv.COLOR_BGR2GRAY)
right_gray=cv.cvtColor(right_color,cv.COLOR_BGR2GRAY)
H=left_gray.shape[0];W=left_gray.shape[1];MaxDis=64
left_cost_volume,right_cost_volume=Census(left_gray,right_gray,MaxDis)
cv.imwrite("CostLL1.png",normalize(np.uint8(np.argmin(left_cost_volume,axis=2)),MaxDis))
cv.imwrite("CostRR1.png",normalize(np.uint8(np.argmin(right_cost_volume,axis=2)),MaxDis))
CadL,CadR=Computer_AD(left_color,right_color,MaxDis)
RawCostL=CostVolume(left_cost_volume,CadL,30.0,10.0,H,W,MaxDis)
RawCostR=CostVolume(right_cost_volume,CadR,30.0,10.0,H,W,MaxDis)
testf=np.uint8(np.argmin(RawCostL,axis=2))
del CadL,CadR,left_cost_volume,right_cost_volume
cv.imwrite("CostL1.png",normalize(np.uint8(np.argmin(RawCostL,axis=2)),MaxDis))
cv.imwrite("CostR1.png",normalize(np.uint8(np.argmin(RawCostR,axis=2)),MaxDis))
print('getting four arm length....')
astart=tt.time()
resultLL=get_arm(left_color,3,1,7,3) #确定左图的ARM
resultRR=get_arm(right_color,3,1,7,3) #确定右图的ARM
end=tt.time()
print('gotted...({:.2f}S)'.format(end-astart))
print('Agging left and right cost...')
test1=aggodd(resultLL,RawCostL)
left_agged=aggodd(resultLL,test1)
#test1=aggodd(resultLL,left_agged)
#left_agged=aggeven(resultLL,test1)

test=aggodd(resultRR,RawCostR)
right_agged=aggodd(resultRR,test)
#test=aggodd(resultRR,right_agged)
#right_agged=aggeven(resultRR,test)

cv.imwrite("leftagg.png",normalize(np.uint8(np.argmin(left_agged,axis=2)),MaxDis))
cv.imwrite("rightagg.png",normalize(np.uint8(np.argmin(right_agged,axis=2)),MaxDis))
"""
aggregation_volume=np.zeros(shape=(H,W,MaxDis,4),dtype=np.float32)
aggtwo,aggfour=agglr(left_agged,left_color,right_color,MaxDis,1.0,3.0,15)
aggone,aggthree=aggtb(left_agged,left_color,right_color,MaxDis,1.0,3.0,15)
aggregation_volume[:,:,:,0]=aggtwo
aggregation_volume[:,:,:,1]=aggfour
aggregation_volume[:,:,:,2]=aggone
aggregation_volume[:,:,:,3]=aggthree
#disparity_left_eight=select_disparity(aggregation_volume)
#cv.imwrite("rrrrr.png",normalize(np.uint8(disparity_left_eight),64))
left_agged=np.sum(aggregation_volume, axis=3)/4.0



aggtwo,aggfour=agglr1(right_agged,left_color,right_color,MaxDis,1.0,3.0,15)
aggone,aggthree=aggtb1(right_agged,left_color,right_color,MaxDis,1.0,3.0,15)
aggregation_volume[:,:,:,0]=aggtwo
aggregation_volume[:,:,:,1]=aggfour
aggregation_volume[:,:,:,2]=aggone
aggregation_volume[:,:,:,3]=aggthree
#disparity_left_eight=select_disparity(aggregation_volume)
#cv.imwrite("rrrrr.png",normalize(np.uint8(disparity_left_eight),64))
right_agged=np.sum(aggregation_volume, axis=3)/4.0
"""
cv.imwrite("leftagged.png",normalize(np.uint8(np.argmin(left_agged,axis=2)),MaxDis))
cv.imwrite("rightagged.png",normalize(np.uint8(np.argmin(right_agged,axis=2)),MaxDis))

left_dis=np.argmin(left_agged,axis=2).astype(np.int32)
testf=left_dis.copy()
right_dis=np.argmin(right_agged,axis=2).astype(np.int32)
final,nor,occ,mis=ClassifyOutlier(left_dis,right_dis,MaxDis)
ggmap=final.copy()
cv.imwrite("lrcheck.png",final)
aaa=tt.time()
for i in range(5):
    left_dis,final=Iterative_Region_Voting(left_dis,final,resultLL,20,0.4,MaxDis)
bbb=tt.time()
print('agssssged...({:.2f}S)'.format(bbb-aaa))
cv.imwrite("Iterative_Region_Voting.png",normalize(left_dis,MaxDis))
cv.imwrite("lrcheckAfter.png",final)
posted=process(final,left_dis)
cv.imwrite("postedC.png",cv.applyColorMap(np.uint8(normalize(posted,MaxDis)), cv.COLORMAP_JET))
cv.imwrite("posted.png",posted)
edge = cv.Canny(np.uint8(posted), 5, 15)
aabb=edge_optimize(edge,posted,left_agged)
cv.imwrite("aabbC.png",cv.applyColorMap(np.uint8(normalize(aabb,MaxDis)), cv.COLORMAP_JET))
ccdd=Sub_pixel_Enhancement(aabb,left_agged)
ccdd=cv.medianBlur(ccdd,3)

diff=np.zeros(shape=(H,W),dtype=np.uint8)
diff[:,:]=255
miserr=0;occerr=0;norerr=0
for h in range(H):
    for w in range(W):
        if np.abs(ref[h,w]-ccdd[h,w])>0.5 and ref[h,w]!=0:
            diff[h,w]=0
cv.imwrite("diff.png",diff)
cv.imshow("origin",left_color)
cv.imshow("diff",diff)
aaend=tt.time()
print('agged...({:.2f}S)'.format(aaend-start))
cv.waitKey()
#print('全部区域错误率:'+str((occerr+miserr+norerr)/(H*W)))
#print('未遮挡区域:'+str(norerr/nor))
ccdd=np.uint8(normalize(ccdd,MaxDis))
cv.imwrite("posted.png",ccdd)
cv.imwrite("postedC.png",cv.applyColorMap(ccdd, cv.COLORMAP_JET))
#aaend=tt.time()
#print('agged...({:.2f}S)'.format(aaend-end))
#end=tt.time()
#print('end...({:.2f}S)'.format(end-start))
#cv.imshow("diff",diff)
#cv.imshow("disparity",ccdd)
#cv.imshow("groundtruth",reff)
#cv.waitKey()
#cv.destroyAllWindows()