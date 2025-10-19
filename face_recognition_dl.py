import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from typing import Tuple, List, Dict
import pickle

class FaceRecognitionDL:
    def __init__(self, model_path: str = 'models', device: str = 'cpu'):
        """Initialize face detection and recognition models."""
        self.device = torch.device(device)
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            device=self.device
        )
        
        # Initialize FaceNet model for face embeddings
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Database of face embeddings
        self.embeddings_file = os.path.join(model_path, 'embeddings.pkl')
        self.embeddings_db: Dict[str, np.ndarray] = {}
        self.load_embeddings()

    def load_embeddings(self):
        """Load saved face embeddings if they exist."""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings_db = pickle.load(f)

    def save_embeddings(self):
        """Save face embeddings to disk."""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings_db, f)

    def get_face_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Convert face image to embedding vector."""
        # Convert to PIL Image if needed
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)
        
        # Get face tensor from MTCNN
        try:
            face_tensor = self.mtcnn(face_img)
            if face_tensor is None:
                raise ValueError("No face detected in image")
                
            # Add batch dimension and move to device
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor).cpu().numpy()
            
            # Normalize embedding
            return normalize(embedding).flatten()
            
        except Exception as e:
            raise ValueError(f"Failed to generate face embedding: {str(e)}")

    def add_face(self, name: str, enrollment: str, image: np.ndarray) -> bool:
        """Add a face to the database."""
        try:
            # Detect face and get embedding
            embedding = self.get_face_embedding(image)
            
            # Store in database
            key = f"{enrollment}_{name}"
            if key not in self.embeddings_db:
                self.embeddings_db[key] = []
            self.embeddings_db[key].append(embedding)
            
            # Save updated database
            self.save_embeddings()
            return True
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            return False

    def identify_face(self, image: np.ndarray, threshold: float = 0.7) -> Tuple[str, float]:
        """Identify a face in the image."""
        try:
            # Get embedding for input face
            query_embedding = self.get_face_embedding(image)
            
            best_match = None
            best_score = -1
            
            # Compare with all stored embeddings
            for key, stored_embeddings in self.embeddings_db.items():
                for embedding in stored_embeddings:
                    score = np.dot(query_embedding, embedding)
                    if score > best_score:
                        best_score = score
                        best_match = key
            
            if best_score >= threshold:
                return best_match, best_score
            return "Unknown", best_score
            
        except Exception as e:
            print(f"Error identifying face: {str(e)}")
            return "Unknown", 0.0

    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces in an image and return their bounding boxes."""
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect faces
        boxes, _ = self.mtcnn.detect(image)
        
        if boxes is None:
            return []
            
        return boxes.astype(int)

    def get_face_locations(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Get face locations in (top, right, bottom, left) format."""
        boxes = self.detect_faces(image)
        locations = []
        
        for box in boxes:
            left, top, right, bottom = box
            locations.append((int(top), int(right), int(bottom), int(left)))
            
        return locations 