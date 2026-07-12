import { createClient } from '@supabase/supabase-js';

// Pulling variables securely from Vite's environment wrangler
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Supabase credentials missing! Check your root .env file configuration.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);