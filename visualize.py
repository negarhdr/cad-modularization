import networkx as nx 
import os 
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import cadquery as cq
import cairosvg

_PATH_ = "/System/Volumes/Data/MyFiles/Postdoc/AIMO/Codes/CADQuery/Test/"
STEP_PATH =  "/System/Volumes/Data/MyFiles/Postdoc/AIMO/Data/SolidLetters/step/"

############################## Visualization ###########################
def crop_img(img_file, image_name):
    img = Image.open(img_file)
    width, height = img.size
    left = 150
    top = 0
    right = 450
    bottom = 4 * height / 4
    img_crp = img.crop((left, top, right, bottom))
    cropped_img_path = os.path.join(_PATH_, "solid_letter_cropped_img/", image_name+".png")
    img_crp.save(cropped_img_path)
    return cropped_img_path

def step_to_image(file_name): 
    
    if (file_name + ".step") in os.listdir(STEP_PATH):
        step_file = os.path.join(STEP_PATH, file_name + ".step")
        solid_ = cq.importers.importStep(step_file)
        assy = cq.Assembly()
        assy.add(solid_, name=file_name)
        # cq.display(assy)
        svg_path = os.path.join(_PATH_, "solid_letter_svg/", file_name+".svg")
        img_path = os.path.join(_PATH_, "solid_letter_img/", file_name+".png")
        cq.exporters.export(assy.toCompound(), svg_path)
        #print('exported')
        cairosvg.svg2png(url=svg_path, write_to=img_path, background_color='#ffffff')
        #print('svg made')
        cropped_img_path = crop_img(img_path, file_name)
        #print('cropped')
        return cropped_img_path
    else:
        print('step file not found')
        return None 
        
def visualize(graph, key='representations'):
    if len(list(graph.edges)) > 0:
        fig, ax = plt.subplots()
        pos = nx.spring_layout(graph, k=3, seed=1)
        # image_path = '/System/Volumes/Data/MyFiles/Postdoc/AIMO/Codes/CADQuery/Test/cropped_img.png'
        for node, (x, y) in pos.items():
            node_font_name = graph.nodes[node][key][1] + "_" + graph.nodes[node][key][2] + "_" + graph.nodes[node][key][3] 
            # print(node_font_name)
            image_path = step_to_image(node_font_name)
            #print(image_path)
            image = np.array(Image.open(image_path))
            #print('image')
            ax.imshow(image, extent=(x - 0.2, x + 0.2, y - 0.2, y + 0.2), zorder=3)
            
        # nx.draw(query_subgraph, with_labels=True, font_weight='bold')
        
        for edge in graph.edges():
            start, end = edge
            start_pos = pos[start]
            end_pos = pos[end]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='black', zorder=1)
        plt.axis('off')
        plt.show()
    else:
        for node in list(graph.nodes):
            node_font_name = graph.nodes[node][key][1] + "_" + graph.nodes[node][key][2] + "_" + graph.nodes[node][key][3] 
            # print(node_font_name)
        print("this retrieved subgraph has no edges")
    # plt.savefig('plot.png')

    return 