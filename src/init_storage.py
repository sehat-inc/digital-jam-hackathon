# It is run only once to inintialize the storage


from supabase import create_client, Client
import os


BUCKET_NAME = 'contract-files'  # Changed bucket name to be more specific
supabase: Client = create_client(
    os.getenv('SERVICE_KEY'),
    os.getenv('ROLE_KEY')
)

def init_storage():
    """Initialize storage bucket with proper configuration"""
    try:
        # List existing buckets
        buckets = supabase.storage.list_buckets()
        bucket_exists = any(bucket['name'] == BUCKET_NAME for bucket in buckets)
        
        if not bucket_exists:
            # Create new bucket with public access
            supabase.storage.create_bucket(
                BUCKET_NAME,
                options={
                    'public': True,  # Allow public access
                    'file_size_limit': 52428800,  # 50MB limit
                    'allowed_mime_types': ['application/pdf']  # Only allow PDFs
                }
            )
            print(f"Created new bucket: {BUCKET_NAME}")
        else:
            print(f"Bucket {BUCKET_NAME} already exists")
            
    except Exception as e:
        print(f"Error initializing storage: {str(e)}")
        raise

# Initialize storage on startup
#print("Initializing storage...")
#init_storage()