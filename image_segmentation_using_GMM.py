

#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


def resize_image(img,Scale):
    """Reshape the image to the scale of the image"""
    #new width
    try:
        width = int(data_img.shape[1] * Scale/ 100)
        
        #new height
        height = int(data_img.shape[0] * Scale/ 100)
        
        #new dimension
        dimension = (width, height)   
        
        # return resized new imag2e
        return cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
    except:
        print("Unable to resize the image in function")
        return None

#variables
Name = input("Enter Image name (without extension): ")
File_Name = Name+".jpg"
Result_Name = Name+"-Segmented.jpg"


#import image
try :
    data_img = cv2.imread("Images/"+File_Name,cv2.IMREAD_COLOR)

except IOError:
    print("File Not exists")

#resized image {User can resize between 1-100% scale of the original image}
new_img = resize_image(data_img,60)



#image resize to lower-dimension for segentation
try:
    res_image = new_img.reshape((-1,3))
except AttributeError:
    print("Unable to resize.")

#implement the model
#coavriance_type can be used a.tied, b.full, c.diag, d. spherical
try: 
    n_comp = int(input("Enter no of components :"))
    seg_model = GMM(n_components=n_comp,covariance_type="full")
    
    #fit the  resahped image
    seg_model.fit(res_image)
    
    #prdicted labels of gmm
    seg_labels = seg_model.predict(res_image)
    print(seg_labels.shape)
    print("\nNo of zeros labels :",len(seg_labels[seg_labels==0]))
    print("\nNo of ones lebels :",len(seg_labels[seg_labels==1]))
    # print(len(seg_labels[seg_labels==2]))
    
    seg_temp = seg_labels
    
    #size of the image
    Shape = new_img.shape
    print("\nShape of the resized image : ",Shape)
    
    image_segmented = seg_labels.reshape(Shape[0],Shape[1])
    
    
    #copy resize image
    for i in range(n_comp):
        temp = res_image.copy()
        for j in range(len(seg_temp)):
            
            if(seg_temp[j]==i):
                temp[j] = np.array([168,168,172]) 
        temp = np.reshape(temp,(Shape[0],Shape[1],3))
        cv2.imshow(Name+"_Mask"+str(i),temp)
        File_Name = Name+"_Mask"+str(i)+".jpg"
        cv2.imwrite("Result/"+File_Name,temp)
    plt.imshow(image_segmented)



    #save file
    plt.tight_layout()
    plt.savefig("Result/"+Result_Name,dpi = 500, bbox_inches = 'tight')
    plt.show()
except:
    print("Unable to create an image.")
#plot



#read stored file
try:
    Data = cv2.imread("Result/"+Result_Name,0)
    Data = resize_image(Data,60)
    # show image
    cv2.imshow("Resized-Tiger",new_img)
    cv2.imshow("Segemented-image",Data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except :
    print("File dont exists")








