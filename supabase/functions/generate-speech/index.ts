import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const fishAudioKey = Deno.env.get('FISH_AUDIO_API_KEY')!

    const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)
    const { text, voiceSampleUrl, userId } = await req.json()

    if (!text || !voiceSampleUrl) {
      throw new Error("Missing required arguments: text or voiceSampleUrl")
    }

    console.log(`Downloading voice reference from storage URL: ${voiceSampleUrl}`)
    const fileResponse = await fetch(voiceSampleUrl)
    if (!fileResponse.ok) throw new Error(`Failed to download voice sample file. Status: ${fileResponse.status}`)
    const referenceAudioBlob = await fileResponse.blob()

    // Create file explicit container wrapper
    const referenceAudioFile = new File([referenceAudioBlob], 'reference.webm', { type: 'audio/webm' })

    const formData = new FormData()
    formData.append('text', text)
    formData.append('model', 's2.1-pro-free') 
    formData.append('reference_audio', referenceAudioFile)

    console.log("Transmitting payload stream directly out to Fish Audio API processing arrays...")

    const ttsResponse = await fetch('https://api.fish.audio/v1/tts', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${fishAudioKey}`
      },
      body: formData
    })

    if (!ttsResponse.ok) {
      const errorMsg = await ttsResponse.text()
      throw new Error(`Fish Audio API returned bad status: ${ttsResponse.status} - ${errorMsg}`)
    }

    const generatedAudioBuffer = await ttsResponse.arrayBuffer()
    const outputFileName = `speech_${Date.now()}.mp3`
    const outputFilePath = `outputs/${outputFileName}`

    console.log("Saving generated voice asset output into storage array lines...")
    const { error: uploadError } = await supabaseAdmin.storage
      .from('generated-audio')
      .upload(outputFilePath, generatedAudioBuffer, {
        contentType: 'audio/mp3',
        cacheControl: '3600'
      })

    if (uploadError) throw uploadError

    const { data: { publicUrl } } = supabaseAdmin.storage
      .from('generated-audio')
      .getPublicUrl(outputFilePath)

    // Wrap db log in an isolated try-catch so it NEVER breaks the audio stream if table doesn't exist
    try {
      await supabaseAdmin
        .from('generation_history')
        .insert([
          {
            user_id: userId || '00000000-0000-0000-0000-000000000000',
            prompt_text: text,
            generated_file_url: publicUrl,
            voice_used: 'Custom Cloned Profile'
          }
        ])
    } catch (dbErr) {
      console.warn("Database log omitted. Create generation_history table to track entries:", dbErr)
    }

    return new Response(
      JSON.stringify({ success: true, audioUrl: publicUrl }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    )

  } catch (error: any) {
    console.error("Internal Runtime Error Path triggered:", error.message)
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    )
  }
})