
import csv
from torchvision.utils import save_image

def save(img,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i):
    
    save_image(img, generated_dir+str(total_generated_img)+'.png')
    save_image(mask, generated_dir+str(total_generated_img)+'_Mask.png')

    with open(generated_csv, 'a') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        
        row = [str(total_generated_img)+'.png', str(total_generated_img)+'_Mask.png', label, intensity]
            
        # writing the data rows 
        csvwriter.writerow(row)
    
        csvfile.close()

def find_label(label,intensity):
    if label == 'Homogeneous':
        lab = 0
    elif label == 'Speckled':
        lab = 1
    elif label == 'Nucleolar':
        lab = 2
    elif label == 'Centromere':
        lab = 3
    elif label == 'Golgi':
        lab = 4
    elif label == 'NuMem':
        lab = 5
    else: # mistp
        lab = 6
        
    if intensity == 'positive':
        inten = 0
    else:
        inten = 1
    
    return lab, inten
