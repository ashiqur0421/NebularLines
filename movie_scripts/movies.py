import os
import re
import ffmpeg

def create_video_from_images(image_folder, output_video, file_pattern, frame_rate=30, resolution=(1920, 1080)):
    """
    Creates a video from a sequence of PNG images based on the provided file pattern.

    :param image_folder: Directory where PNG images are located.
    :param output_video: Output video file (e.g., 'output_intensity.mp4').
    :param file_pattern: Pattern used to identify image files for the sequence (e.g., 'intensity', 'density').
    :param frame_rate: Frame rate for the output video.
    :param resolution: Resolution for the output video (width, height).
    """

    # List all PNG files in the directory and filter by the pattern
    images = [img for img in os.listdir(image_folder) if img.endswith('.png') and file_pattern in img]
    
    if not images:
        raise ValueError(f"No images matching the pattern '{file_pattern}' found in the folder.")

    # Sort images numerically by the index extracted from the filename
    images.sort(key=lambda x: int(re.search(r'(\d+)', x).group(0)))

    # Build FFmpeg input pattern for images
    input_pattern = os.path.join(image_folder, f"{file_pattern}%05d.png")  # Assumes image naming like output_00319...

    # Command to convert images to a video
    ffmpeg.input(input_pattern, framerate=frame_rate).output(
        output_video,
        video_bitrate='5000k',
        s=f'{resolution[0]}x{resolution[1]}',  # Set the resolution
        pix_fmt='yuv420p'  # Ensure compatibility with most players
    ).run()

    print(f"Video created successfully: {output_video}")


def create_movies_for_sequences(image_folder, frame_rate=30, resolution=(1920, 1080)):
    """
    Create a separate movie for each unique sequence of images based on their filename patterns.
    :param image_folder: Directory containing the image sequences.
    :param frame_rate: Frame rate for the output video.
    :param resolution: Resolution for the output video (width, height).
    """

    # Identify unique patterns in the filenames to create separate movies for each type
    all_images = os.listdir(image_folder)
    
    # Use regular expressions to group files by their unique patterns
    intensity_pattern = "intensity"
    density_pattern = "density"
    
    # Process each pattern separately
    if any(intensity_pattern in img for img in all_images):
        print(f"Processing images with pattern '{intensity_pattern}'")
        create_video_from_images(image_folder, f"intensity_movie.mp4", intensity_pattern, frame_rate, resolution)
    
    if any(density_pattern in img for img in all_images):
        print(f"Processing images with pattern '{density_pattern}'")
        create_video_from_images(image_folder, f"density_movie.mp4", density_pattern, frame_rate, resolution)


# Example usage:
image_folder = 'path/to/your/images'
frame_rate = 30  # 30 frames per second
resolution = (1920, 1080)  # Resolution of the output video

create_movies_for_sequences(image_folder, frame_rate, resolution)