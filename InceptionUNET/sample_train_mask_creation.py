from lxml import etree
import numpy as np
from PIL import Image, ImageDraw
import glob
from skimage.transform import resize
from skimage.util import img_as_bool
import multiprocessing as mp
import matplotlib.pyplot as plt

def parse(file):  # parses the xhtml file
    tree = etree.parse(file)
    root_ = tree.getroot()
    return root_


def draw(layer, boundings):
    layer = Image.fromarray(layer, '1')
    drawing = ImageDraw.Draw(layer, '1')
    for poly in boundings:
        try:
            drawing.polygon(poly, fill=1)
        except TypeError:
            pass

    return np.array(layer)


root_folder = '/content/drive/My Drive/NLP-OCR/PRImA Layout Analysis Dataset'
folder = '/NPmasks_384_256b/'

files = glob.glob(root_folder + '/XML/*.xml' )
num = len(files)
print(num)

#for obtaining segementation masks from the xml data

def converter(a,b):
    for i,file in enumerate(files[a:b]):
        print(i)
        with open(file, 'r') as xml:
            root = parse(xml)

            metadata = root[0]
            page = root[1]

            imgname = page.attrib['imageFilename']
            h, w = int(page.attrib['imageHeight']), int(page.attrib['imageWidth'])

            box_dict = {

                "paragraph": [],
                "drop-capital": [],
                "heading": [],
                "caption": [],
                "header": [],
                "footer": [],
                "page-number": [],
                "graphic": [],
                "table": [],
                "footnote": [],
                "footnote-continued": [],
                "credit": [],
                "catch-word": [],
                "signature-mark": [],
                "TOC-entry": [],
                "floating": [],
                "marginalia": [],
                "noise": [],
            }

            final_dict= {
                "paragraph": [], 
                "heading": [],
                "floating":[],
                "caption": [],
                "header-footer": [], 
                "table":[],
                "graphic": [],
                
            }

            for region in page:
                mask_list = []
                for coord in region[0]:
                    mask_list.append((int(coord.attrib['x']), int(coord.attrib['y'])))
                try:
                    if region.tag.split('}')[1] == 'TextRegion':
                        box_dict[region.attrib['type']].append(mask_list)

                    elif region.tag.split('}')[1] == 'MathsRegionType':
                        box_dict['paragraph'].append(mask_list)

                    elif region.tag.split('}')[1] == 'TableRegion':
                        box_dict['table'].append(mask_list)

                    elif region.tag.split('}')[1] == 'ImageRegion' \
                        or region.tag.split('}')[1] == 'LineDrawingRegion'\
                        or region.tag.split('}')[1] == 'ChartRegion'\
                        or region.tag.split('}')[1] == 'GraphicRegion' :
                        box_dict['graphic'].append(mask_list)
                except KeyError:
                    print('KeyError' + file )
                    pass

                final_dict['paragraph'] = box_dict['paragraph'] + box_dict['credit']  + box_dict['TOC-entry'] + box_dict['drop-capital']
                final_dict['heading']  = box_dict['heading']
                # final_dict['drop-capital'] = box_dict['drop-capital']
                final_dict['floating'] = box_dict['floating']
                final_dict['caption'] = box_dict['caption']
                final_dict['header-footer'] = box_dict['header'] + box_dict['footer'] + box_dict['page-number'] + box_dict['footnote'] \
                                            + box_dict['footnote-continued'] + box_dict['catch-word'] 
                final_dict['table'] = box_dict['table']
                final_dict['graphic'] = box_dict['graphic'] + box_dict['signature-mark'] + box_dict['marginalia']
                
                
            try:
                mask = np.zeros((h, w,8), dtype=np.bool)
            except ValueError:
                print('ValueError' + file )
                continue

            for i, key in enumerate(final_dict):
                    mask[:, :, i] = draw(mask[:, :, i], final_dict[key])

            
            

        mask = resize(mask,(384, 256),anti_aliasing=True, ) #1152*768
        mask = mask.astype('bool')
        backg = np.max(mask, axis=2)
        mask[:,:,7] = np.logical_not(backg)

        plt.imshow(mask[:,:,7])
        plt.show()
        np.save(root_folder + folder + imgname.split('.')[0], mask)


# for converting the training images to the same size

root_folder = '/content/drive/My Drive/NLP-OCR/PRImA Layout Analysis Dataset'
img_files = glob.glob(root_folder + '/Images/*.tif' )
num = len(img_files)
print(num)
def img_converter(a,b):
    for i,img_n in enumerate(img_files[a:b]):
        imag = Image.open(img_n).convert('RGB')
        imag = imag.resize((256,384),Image.LANCZOS)#512,768
        save_path = img_n.replace('Images','Images_384_256').replace('.tif','.png')
        imag = imag.save(save_path)
        print(i)
