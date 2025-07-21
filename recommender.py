import os
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec

# Define local dataset path
DATASET_CLOTH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../clothes_tryon_dataset/train/cloth"))

# Load FashionCLIP
model_id = "patrickjohncyh/fashion-clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Pinecone API Key and config
os.environ["PINECONE_API_KEY"] = "pcsk_6gzckn_fSn2dSM451QCCDKMtn5beikgzBcapR3S5mAWxSdHDA44zz2GwcAPBMfnwHEbf9"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "fashion-clip-index"
cloud = "aws"
region = "us-east-1"  # your region
dimension = 512

# Create index if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = pc.Index(index_name)

# Load and preprocess images
def load_images_from_folder(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

# Embed and upload
def embed_and_upload(image_folder):
    image_paths = load_images_from_folder(image_folder)
    batch = []

    for i, img_path in enumerate(image_paths):
        file_name = os.path.basename(img_path)

        pixel_values = preprocess_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values)
            image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)

        embedding = image_embeds.squeeze().cpu().tolist()

        batch.append({"id": file_name, "values": embedding})

        # Upload every 100 or final
        if len(batch) == 50 :
            index.upsert(vectors=batch)
            print(f"Uploaded {len(batch)} embeddings to Pinecone.")
            batch = []
    if batch:
        index.upsert(vectors=batch)
        print(f"✅ Uploaded final batch of {len(batch)} to Pinecone.")

# Use local dataset path
def main():
    image_folder_path = DATASET_CLOTH_DIR
    embed_and_upload(image_folder_path)

    stats = index.describe_index_stats()
    print(stats)

    # Example fetch by filename (not full path)
    example_id = os.listdir(image_folder_path)[0]
    response = index.fetch(ids=[example_id])
    print(response)

    def show_image_matches(matches, base_path):
        for match in matches:
            img_path = os.path.join(base_path, match["id"])
            print(f"{match['id']} | Score: {match['score']}")
            if os.path.exists(img_path):
                display(Image.open(img_path).resize((200, 200)))

    def image_to_vector(image_path):
        img = preprocess_image(image_path).unsqueeze(0).to(device)
        with torch.no_grad():
            img_embed = model.get_image_features(img)
            img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=-1)
        return img_embed.squeeze().cpu().tolist()

    # Example query
    query_image = os.listdir(image_folder_path)[0]
    query_vector = image_to_vector(os.path.join(image_folder_path, query_image))
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    show_image_matches(results["matches"], base_path=image_folder_path)

if __name__ == "__main__":
    main()
