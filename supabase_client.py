# supabase_client.py
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL/KEY missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
