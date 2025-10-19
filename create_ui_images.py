from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(text, filename, size=(200, 200), bg_color=(28, 28, 28), text_color=(255, 255, 0)):
    # Create a new image with a dark background
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to create text
    try:
        # Use a default system font
        font_size = 30
        font = ImageFont.load_default()
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate text position (center)
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=text_color, font=font)
        
    except Exception as e:
        print(f"Error creating text for {filename}: {str(e)}")
    
    # Save the image
    img.save(os.path.join('UI_Image', filename))

# Create UI images
if not os.path.exists('UI_Image'):
    os.makedirs('UI_Image')

# Create logo
logo = Image.new('RGB', (50, 47), (28, 28, 28))
logo.save(os.path.join('UI_Image', '0001.png'))

# Create other UI elements
create_icon("Register", "register.png")
create_icon("Attendance", "attendance.png")
create_icon("Verify", "verifyy.png")

print("UI images created successfully!") 