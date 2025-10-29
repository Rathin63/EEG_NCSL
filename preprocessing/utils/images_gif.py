from PIL import Image
import glob
import moviepy.editor as mp

location = 'C:\\Users\\adaraie\\Desktop\\NCSL_Desk\\Prediction\\Codes\\Python\\PY23N011\\IIEoI_2d_plots'
output_name = 'IIEoI_2d_plots'
duration = 500
loop = 50 # 0: infinite, N: N times loop

patient_iD = 'PY23N011'

# # Create the frames
frames = []
imgs = glob.glob(location+"\\*.png")
for i in imgs:
    image = Image.open(i)
    new_image = image
    new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
    new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
    frames.append(new_image)

# Save into a GIF file that loops forever
frames[0].save(f'{location}\\{patient_iD}_{output_name}_duration-{duration}_loop-{loop}.gif', format='gif',
               append_images=frames[1:],
               save_all=True,
               duration=duration, loop=1,
                dpi=(300, 300))
