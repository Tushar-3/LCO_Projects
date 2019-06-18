# Road_Lane_Detecttion
**Dependencies**
- Python
- Opencv-Pyhton
- numpy
- matplotlib


#### Orignal Image
![Original](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img1.png)

### Step-1:Grayscale
In very first step we have to convet the RGB image into GRAY Scale, then extract the specific color throug it.

![Gray](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img2.png)

### Step-2:Gaussian Blur
To get good result, we have make little bit blur the grascale image.

![Gaussian Blur](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img3.png)

### Step-3:Canny Edge Detection
In this step we will find the edges using Canny algorithum which is alredy provided by Opencv.In canny detction it's looks for sudden chage of color in horizonaly and vertically both.

![canny](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img4.png)

### Step-4:Draw line
In this step we are drawing red line over the white lane.

![Dwar](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img5.png)


### Step-5:Extract usefull area
In this step we are removing the extra stuff from the image with the help of masking technique.

    #vertex for drawing apolygen on the black image
    ver=get_vertice(image)
    

![mask](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img6.png)

  
    #get region of interest
    roi=region_of_interest(edge_image,ver)
    


![mask-lane](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img7.png)

### Step-6:Bit-wise and
In this step we are doing bitwise and with img_canny,mask.

![And](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img8.png)

### Step-7:Image apply
![IMG](https://raw.githubusercontent.com/Tushar-3/Road_Lane_Detecttion/master/Image/img9.png)
