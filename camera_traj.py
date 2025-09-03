import numpy as np
import sapien.core as sapien
from typing import List


def generate_circular_trajectory(center: np.ndarray, radius: float, 
                               height: float, n_frames: int, 
                               full_rotation: bool = True) -> List[sapien.Pose]:
    """
    Generate circular camera trajectory around a center point.
    
    Args:
        center: Center point to orbit around
        radius: Radius of the circular path
        height: Height offset from the center
        n_frames: Number of frames/poses to generate
        full_rotation: Whether to complete a full 360° rotation
        
    Returns:
        List of camera poses
    """
    poses = []
    angle_step = (2 * np.pi if full_rotation else np.pi) / n_frames
    
    for i in range(n_frames):
        angle = i * angle_step
        
        # Camera position
        cam_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
        ])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        
        # Right vector  
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality
        up = np.cross(right, forward)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([forward, -right, up])
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = cam_pos
        
        pose = sapien.Pose.from_transformation_matrix(transform)
        poses.append(pose)
        
    return poses


def generate_sphere_spiral_trajectory(center: np.ndarray, radius: float,
                                     start_elevation: float, end_elevation: float,
                                     rotations: float, n_frames: int) -> List[sapien.Pose]:
    """
    Generate sphere spiral camera trajectory.
    
    Args:
        center: Center point of the sphere
        radius: Radius of the sphere
        start_elevation: Starting elevation angle in degrees (-90 to 90)
        end_elevation: Ending elevation angle in degrees (-90 to 90)
        rotations: Number of horizontal rotations
        n_frames: Number of frames/poses to generate
        
    Returns:
        List of camera poses
    """
    poses = []
    
    # Convert elevation angles from degrees to radians
    start_elev_rad = np.radians(start_elevation)
    end_elev_rad = np.radians(end_elevation)
    
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0  # 0 to 1
        
        # Interpolate elevation angle
        elevation = start_elev_rad + t * (end_elev_rad - start_elev_rad)
        
        # Calculate azimuth angle (horizontal rotation)
        azimuth = t * rotations * 2 * np.pi
        
        # Convert spherical coordinates to Cartesian
        # elevation: angle from horizontal plane (0 = horizontal, π/2 = up, -π/2 = down)
        # azimuth: angle around vertical axis
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        
        cam_pos = center + np.array([x, y, z])
        
        # Camera always looks at center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Calculate up vector (tangent to sphere surface, pointing generally upward)
        # For a sphere, the up vector is perpendicular to the radial direction
        radial = cam_pos - center
        radial = radial / np.linalg.norm(radial)
        
        # Use world up as reference
        world_up = np.array([0, 0, 1])
        
        # Calculate right vector
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:  # Handle singularity when looking straight up/down
            # Use a different reference vector
            right = np.cross(forward, np.array([1, 0, 0]))
        right = right / np.linalg.norm(right)
        
        # Recompute up vector to ensure orthogonality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix (SAPIEN camera convention: X=forward, Y=left, Z=up)
        rotation_matrix = np.column_stack([forward, -right, up])
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = cam_pos
        
        pose = sapien.Pose.from_transformation_matrix(transform)
        poses.append(pose)
        
    return poses
