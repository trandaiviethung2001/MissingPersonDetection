"""
Face recognition - match faces against missing person database.
Uses InsightFace ArcFace 512-d embeddings with cosine similarity.
"""
import os
import pickle
import numpy as np
import cv2
from insightface.app import FaceAnalysis


def _safe_normalize(vec):
    """L2-normalize a 1-D embedding, returning None if norm is zero."""
    norm = np.linalg.norm(vec)
    if norm <= 0 or not np.isfinite(norm):
        return None
    return vec / norm


class FaceRecognizer:
    """Match face embeddings against a missing person database."""

    def __init__(self, embeddings_path, threshold=0.4):
        """
        Args:
            embeddings_path: Path to embeddings.pkl file.
            threshold: Minimum cosine similarity to consider a match.
        """
        self.threshold = threshold
        self.known_embeddings = []
        self.known_names = []
        self.known_ids = []

        if os.path.exists(embeddings_path):
            self._load_database(embeddings_path)
            print(f"[FaceRecognizer] Loaded {len(self.known_embeddings)} embeddings "
                  f"for {len(set(self.known_names))} missing persons.")
        else:
            print(f"[FaceRecognizer] WARNING: Embeddings file not found at {embeddings_path}")
            print("  Run train_face_recognition.ipynb to build the database.")

    def _load_database(self, path):
        """Load embeddings database from pickle file."""
        with open(path, "rb") as f:
            database = pickle.load(f)

        for person in database:
            person_id = person["person_id"]
            name = person["name"]
            embeddings = person["embeddings"]

            for emb in embeddings:
                norm_emb = _safe_normalize(np.asarray(emb, dtype=np.float32))
                if norm_emb is None:
                    print(f"[FaceRecognizer] WARNING: skipping zero-norm embedding for {person_id}")
                    continue
                self.known_embeddings.append(norm_emb)
                self.known_names.append(name)
                self.known_ids.append(person_id)

        self.known_embeddings = np.array(self.known_embeddings) if self.known_embeddings else np.array([])

    def match(self, embedding):
        """
        Match a face embedding against the missing person database.

        Args:
            embedding: 512-d numpy array (ArcFace embedding).

        Returns:
            Tuple (name, similarity, person_id):
                - name: Matched person name or None
                - similarity: Highest cosine similarity
                - person_id: Matched person ID or None
        """
        if len(self.known_embeddings) == 0:
            return None, 0.0, None

        norm_emb = _safe_normalize(np.asarray(embedding, dtype=np.float32))
        if norm_emb is None:
            return None, 0.0, None
        similarities = np.dot(self.known_embeddings, norm_emb)
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])

        if max_similarity >= self.threshold:
            return self.known_names[max_idx], max_similarity, self.known_ids[max_idx]

        return None, max_similarity, None

    def match_raw(self, embedding):
        """
        Match without threshold — always returns the best-matching person.

        Used by the tracker's state machine which applies its own thresholds.

        Returns:
            Tuple (name, similarity, person_id) — never (None, ..., None).
            Returns (None, 0.0, None) only when the database is empty.
        """
        if len(self.known_embeddings) == 0:
            return None, 0.0, None

        norm_emb = _safe_normalize(np.asarray(embedding, dtype=np.float32))
        if norm_emb is None:
            return None, 0.0, None
        similarities = np.dot(self.known_embeddings, norm_emb)
        max_idx = int(np.argmax(similarities))
        return (self.known_names[max_idx],
                float(similarities[max_idx]),
                self.known_ids[max_idx])

    @staticmethod
    def build_database(db_dir, output_path, app=None):
        """
        Build embeddings database from a directory of missing person photos.

        Directory structure:
            db_dir/
                person_001/
                    name.txt (optional)
                    photo1.jpg
                    photo2.jpg
                person_002/
                    photo1.jpg
                ...

        Args:
            db_dir: Root directory containing person photos.
            output_path: Output .pkl file path.
            app: InsightFace FaceAnalysis app (creates new if None).

        Returns:
            List[dict]: The built database.
        """
        if app is None:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))

        database = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        person_dirs = sorted([
            d for d in os.listdir(db_dir)
            if os.path.isdir(os.path.join(db_dir, d)) and d.startswith("person_")
        ])

        print(f"Found {len(person_dirs)} persons in database.")

        for person_dir in person_dirs:
            person_path = os.path.join(db_dir, person_dir)
            person_id = person_dir

            # Read name from name.txt if available
            name_file = os.path.join(person_path, "name.txt")
            if os.path.exists(name_file):
                with open(name_file, "r", encoding="utf-8") as f:
                    name = f.read().strip()
            else:
                name = person_dir

            embeddings = []
            image_files = [
                f for f in os.listdir(person_path)
                if os.path.splitext(f)[1].lower() in image_extensions
            ]

            print(f"\n  [{person_id}] {name}: {len(image_files)} images")

            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"    WARNING: Cannot read image {img_file}")
                    continue

                faces = app.get(image)

                if len(faces) == 0:
                    print(f"    WARNING: No face found in {img_file}")
                elif len(faces) > 1:
                    # Pick the largest face
                    areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
                    best_face = faces[np.argmax(areas)]
                    print(f"    WARNING: Found {len(faces)} faces in {img_file}, using largest.")
                    embeddings.append(best_face.embedding)
                    print(f"    OK: {img_file}")
                else:
                    embeddings.append(faces[0].embedding)
                    print(f"    OK: {img_file}")

            if len(embeddings) == 0:
                print(f"    ERROR: No embeddings for {person_id}!")
                continue

            database.append({
                "person_id": person_id,
                "name": name,
                "embeddings": embeddings
            })

        # Save
        with open(output_path, "wb") as f:
            pickle.dump(database, f)

        total_embeddings = sum(len(p["embeddings"]) for p in database)
        print(f"\nSaved database: {len(database)} persons, {total_embeddings} embeddings")
        print(f"File: {output_path}")

        return database
