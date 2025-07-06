"""
Video creation utilities for generating videos from image sequences
"""

import os
import logging
import subprocess
import glob


def create_video(input_dir, image_pattern, output_video, fps=10, quality='high'):
    """
    Create a video from a sequence of images using ffmpeg
    
    Args:
        input_dir: Directory containing the images
        image_pattern: Pattern for input images (e.g., "%04d_rgb_gt.png")
        output_video: Output video file path
        fps: Frames per second
        quality: Video quality ('high', 'medium', 'low')
    """
    
    # Quality settings
    quality_settings = {
        'high': ['-crf', '18', '-preset', 'slow'],
        'medium': ['-crf', '23', '-preset', 'medium'], 
        'low': ['-crf', '28', '-preset', 'fast']
    }
    
    # Convert to absolute paths before changing directory
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_video = os.path.abspath(output_video)
    
    # Change to input directory for ffmpeg patterns to work
    original_cwd = os.getcwd()
    os.chdir(abs_input_dir)
    
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-i', image_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            *quality_settings[quality],
            abs_output_video
        ]
        
        logging.info(f"Creating video: {abs_output_video}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"✓ Successfully created {abs_output_video}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Error creating {abs_output_video}: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Skipping video creation.")
        return False
    finally:
        os.chdir(original_cwd)


def create_side_by_side_video(input_dir, gt_pattern, out_pattern, output_video, fps=10, quality='high'):
    """
    Create a side-by-side comparison video
    
    Args:
        input_dir: Directory containing the images
        gt_pattern: Pattern for GT images
        out_pattern: Pattern for output images  
        output_video: Output video file path
        fps: Frames per second
        quality: Video quality
    """
    
    quality_settings = {
        'high': ['-crf', '18', '-preset', 'slow'],
        'medium': ['-crf', '23', '-preset', 'medium'], 
        'low': ['-crf', '28', '-preset', 'fast']
    }
    
    # Convert to absolute paths before changing directory
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_video = os.path.abspath(output_video)
    
    # Change to input directory for ffmpeg patterns to work
    original_cwd = os.getcwd()
    os.chdir(abs_input_dir)
    
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-i', gt_pattern,
            '-framerate', str(fps), 
            '-i', out_pattern,
            '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            *quality_settings[quality],
            abs_output_video
        ]
        
        logging.info(f"Creating side-by-side video: {abs_output_video}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"✓ Successfully created {abs_output_video}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Error creating {abs_output_video}: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.warning("ffmpeg not found. Skipping video creation.")
        return False
    finally:
        os.chdir(original_cwd)


def create_videos_from_images(output_dir, fps=10, quality='high', create_comparison=True):
    """
    Create videos from all generated images
    
    Args:
        output_dir: Directory containing the generated images
        fps: Frames per second for the video
        quality: Video quality ('high', 'medium', 'low')
        create_comparison: Whether to create side-by-side comparison videos
    """
    logging.info("Creating videos from generated images...")

    # Check what types of RGB images we have (excluding depth)
    image_types = []
    for suffix in ['rgb_gt', 'rgb_out', 'rgb_cond']:
        pattern = os.path.join(output_dir, f"*_{suffix}.png")
        if glob.glob(pattern):
            image_types.append(suffix)

    logging.info(f"Found RGB image types: {image_types}")

    if not image_types:
        logging.warning("No RGB images found for video creation!")
        return

    # Create individual videos
    success_count = 0
    for img_type in image_types:
        pattern = f"%04d_{img_type}.png"
        output_video = os.path.join(output_dir, f"{img_type}_video.mp4")

        if create_video(output_dir, pattern, output_video, fps, quality):
            success_count += 1

    logging.info(f"✓ Created {success_count}/{len(image_types)} individual RGB videos")

    # Create comparison videos if requested (only RGB comparison)
    if create_comparison:
        comparisons = [
            ('rgb_gt', 'rgb_out', 'rgb_comparison'),
        ]

        comparison_count = 0
        for gt_type, out_type, comp_name in comparisons:
            if gt_type in image_types and out_type in image_types:
                gt_pattern = f"%04d_{gt_type}.png"
                out_pattern = f"%04d_{out_type}.png"
                output_video = os.path.join(output_dir, f"{comp_name}_video.mp4")

                if create_side_by_side_video(output_dir, gt_pattern, out_pattern, output_video, fps, quality):
                    comparison_count += 1

        logging.info(f"✓ Created {comparison_count} RGB comparison videos")
    # Remove all rgb_gt and rgb_out files
    for suffix in ["rgb_gt", "rgb_out"]:
        pattern = os.path.join(output_dir, f"*_{suffix}.png")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                logging.info(f"Removed file: {file_path}")
            except Exception as e:
                logging.warning(f"Could not remove file {file_path}: {e}")
    logging.info(f"🎬 All videos saved to: {output_dir}")
