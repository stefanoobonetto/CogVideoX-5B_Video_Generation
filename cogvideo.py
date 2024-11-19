import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

# Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
model_id = "THUDM/CogVideoX-5b"

# Thank you [@camenduru](https://github.com/camenduru)!
# The reason for using checkpoints hosted by Camenduru instead of the original is because they exported
# with a max_shard_size of "5GB" when saving the model with `.save_pretrained`. The original converted
# model was saved with "10GB" as the max shard size, which causes the Colab CPU RAM to be insufficient
# leading to OOM (on the CPU)

transformer = CogVideoXTransformer3DModel.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="transformer", torch_dtype=torch.float16)
text_encoder = T5EncoderModel.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="text_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

# Create pipeline and run inference
pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)

pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_tiling()

plain_prompts = {
    # "Soldier" : "A sad soldier during war.",
    "Soldier" : "A weary soldier, clad in a dusty, camouflage uniform, stands solemnly in front of the camera, his eyes reflecting a deep sadness and resignation. His face, marked by the grime of battle and the weight of impending conflict, conveys a poignant awareness that the war is imminent. The background is a blur of military activity, hinting at the chaos about to unfold. His posture is rigid yet somehow defeated, as he clutches his helmet in one hand, a symbol of the protection and burden he carries. The somber lighting casts shadows over his features, emphasizing the heavy toll of his duty and the somber realization that his time to face the horrors of war has arrived.",
    "Ball" : "A small, brightly colored rubber ball bounces rhythmically in slow motion against a wooden floor in a sunlit room. Each time the ball hits the floor, it flattens slightly before springing back into its round shape, reaching different heights with each bounce. The camera is set close to the floor, capturing the textures of the wood grain and the way sunlight highlights the ball’s surface. As it bounces, shadows shift dynamically around it, creating a sense of depth and movement.",
    "Car" : "A sleek, black sports car glides effortlessly through a vibrant cityscape at dusk, its polished surface reflecting the neon lights and towering skyscrapers. The scene transitions to a close-up of the car's grille, highlighting its intricate design and glowing headlights. Next, the camera pans to the driver, a confident man in a stylish leather jacket, his hands gripping the steering wheel with ease. The car accelerates smoothly, leaving a trail of soft, glowing taillights against the backdrop of a starlit night, capturing the essence of luxury and speed.",
    "Purse" : "A sleek, black leather purse rests elegantly on a polished wooden table, its surface reflecting a soft, ambient light. The purse features a delicate gold chain strap and a minimalist, rectangular clasp adorned with a subtle, sparkling gem. As the camera zooms in, the texture of the leather becomes apparent, showcasing its fine grain and supple feel. The background gradually shifts to reveal a luxurious, dimly lit room with velvet curtains and a crystal chandelier, enhancing the purse's aura of sophistication and style. The final shot captures the purse from above, highlighting its compact yet spacious design, perfect for an evening of elegance.",
    "TV" : "A sleek, modern living room comes to life as the central focus shifts to a state-of-the-art, ultra-thin television mounted on a pristine white wall. The TV screen flickers to life, displaying a vibrant, high-definition scene of a bustling cityscape at dusk, with neon lights reflecting off skyscraper windows. Surrounding the TV, the room is adorned with contemporary furniture: a plush gray sofa, a glass coffee table, and minimalist decor. Soft, ambient lighting casts a warm glow, highlighting the contrast between the dynamic on-screen imagery and the serene, inviting atmosphere of the room. The scene captures the perfect blend of technology and comfort, inviting viewers to immerse themselves in the ultimate home entertainment experience.",
    "Sofa" : "A plush, vintage sofa with intricate wooden carvings and deep, velvety red upholstery stands prominently in a cozy, dimly lit living room. The sofa's rich texture contrasts with the warm, golden glow of a nearby floor lamp, casting soft shadows on the walls adorned with eclectic art pieces. A stack of well-read books rests on one arm, while a hand-knitted throw drapes over the back, adding a homely touch. The scene is further enhanced by a small, antique wooden table beside the sofa, holding a steaming cup of tea and a vintage clock ticking gently, creating an inviting atmosphere of comfort and nostalgia.",
    "Bike" : "A sleek, yellow mountain bike stands poised on a rugged, forested trail, its shiny frame reflecting the dappled sunlight filtering through the dense canopy. The scene transitions to a close-up of the bike's intricate gear system, showcasing the precision engineering. Next, a pair of hands in padded gloves grips the handlebars, preparing for a journey. The bike then moves into action, navigating over gnarled roots and rocky terrain, the tires kicking up a spray of dirt. Finally, the bike comes to a rest on a hilltop, overlooking a breathtaking panoramic view of the lush, green valley below, symbolizing the freedom and adventure of cycling.",
    "Horse" : "A majestic black stallion gallops through a sun-drenched meadow, its mane flowing like a dark wave, muscles rippling beneath its sleek coat. The scene shifts to a close-up of the horse's intense, intelligent eyes, reflecting the vibrant green of the surrounding grass. As the stallion rears up, its powerful legs silhouetted against the bright sky, a sense of freedom and strength emanates. The background transitions to a panoramic view of rolling hills and a distant forest, emphasizing the horse's solitary grandeur. Finally, the stallion slows to a gentle trot, its breath visible in the cool morning air, embodying the essence of wild elegance.",
    "Cow" : "A majestic cow stands gracefully in a lush, green meadow, its sleek black coat shimmering under the golden rays of the setting sun. The cow's gentle eyes reflect a sense of serenity, as it gazes peacefully at the horizon. Around it, wildflowers sway gently in the breeze, adding vibrant splashes of color to the scene. The cow's long, elegant lashes catch the light, highlighting its calm and noble presence. In the background, a distant oak tree provides a picturesque silhouette, completing this tranquil snapshot of rural beauty.",
    "Thunder" : "The scene opens with a dark, turbulent sky, heavy with storm clouds. Suddenly, a jagged bolt of lightning pierces through, illuminating the landscape for a brief moment in a blinding flash of white and blue. The rumble of thunder follows, echoing deeply, shaking the ground and resonating with an intense, raw energy. Rain begins to fall in heavy, visible drops, splashing against the ground, while the dim light flickers, casting eerie shadows against the backdrop of a windswept, stormy sky.",
    "Dancer" : "A ballet dancer performs an elegant pirouette on a stage with dim lighting, with a single spotlight illuminating her. She spins gracefully, extending one leg and raising her arms, creating beautiful, flowing movements with her dress billowing around her. The camera slowly pans around her, capturing her graceful expression and focused gaze. The background fades into darkness, putting full focus on her form and movements as she completes the pirouette, transitioning into a smooth arabesque pose, with the spotlight creating a dramatic silhouette effect.",
    "Skier" : "A skier descends a snowy mountain slope at high speed, kicking up powder with each turn. The camera follows closely from behind, capturing the rush of snow and the bright sunlight reflecting off the white landscape. As the skier carves through the snow, you can see the details of their ski gear, the stripes on their helmet, and the motion of their poles slicing through the air. Trees in the background are blurred, emphasizing the speed and intensity of the descent, while the skier performs a quick jump over a small ridge, landing smoothly and continuing down the mountain.",
    "Butterfly" : "A delicate butterfly with iridescent blue and purple wings flutters gracefully above a vibrant meadow, its wings shimmering in the golden sunlight. The scene transitions to a close-up of the butterfly's intricate wing patterns, revealing tiny scales that catch the light. Next, the butterfly lands gently on a bright pink flower, its proboscis extending to sip nectar. The background features a lush green landscape with wildflowers in full bloom, creating a serene and picturesque setting. The butterfly then takes flight again, soaring through the air with effortless elegance, against a backdrop of a clear blue sky dotted with fluffy white clouds.",
    "Mouse" : "A tiny, agile mouse with soft, brown fur and bright, curious eyes scurries through a lush, green forest. It pauses to nibble on a sunlit seed, its whiskers twitching inquisitively. The mouse then darts across a fallen log, its delicate paws barely making a sound on the mossy surface. As it explores, the mouse encounters a vibrant butterfly, causing it to momentarily freeze in awe. The scene shifts to the mouse burrowing into a cozy, leaf-lined nest, its safe haven amidst the towering trees and dappled sunlight.",
    "Crocodile" : "A massive crocodile basks on a riverbank, its scaly, olive-green skin glistening under the tropical sun. Its powerful jaws are slightly agape, revealing rows of sharp, white teeth. The creature's eyes, small but piercing, survey the surroundings with a blend of lethargy and alertness. The river flows gently beside it, reflecting the lush greenery of the jungle canopy above. As the scene shifts, the crocodile slides into the water with a graceful, yet menacing ease, creating ripples that disturb the serene surface, all while maintaining its regal and formidable presence in the heart of the wild.",
    "Farmer" : "A rugged farmer, clad in a weathered denim jacket, dusty overalls, and a wide-brimmed straw hat, stands amidst a golden wheat field at sunset. His calloused hands gently cradle a freshly harvested ear of corn, his face etched with lines of wisdom and resilience. The warm, amber glow of the setting sun bathes the scene, casting long shadows and highlighting the rich, earthy tones of the soil. As he looks out over his sprawling farm, a sense of pride and connection to the land emanates from his weathered features, embodying the timeless spirit of agriculture.",
    "Pilot" : "A seasoned pilot, clad in a sleek black flight suit with a white stripe down the leg, stands confidently in a vast, dimly lit hangar. His aviator sunglasses reflect the soft glow of overhead lights, and his hands rest casually on his hips. The scene shifts to show him striding purposefully towards a state-of-the-art fighter jet, its metallic surface gleaming under the hangar's lights. As he approaches, he reaches up to adjust his pilot's cap, revealing a hint of a determined smile. The background features rows of advanced aircraft and bustling ground crew, emphasizing the high-stakes atmosphere of a military airbase. The pilot's posture exudes authority and readiness, capturing the essence of a skilled aviator preparing for a critical mission.",
    "TaxiDriver" : "A seasoned taxi driver, sporting a classic black cap and a navy blue jacket, navigates the bustling city streets with a focused yet weary expression. The interior of his vintage yellow cab is filled with small trinkets and a worn-out map, reflecting years of service. As he drives, the cityscape whirls past, showcasing neon signs and towering skyscrapers. A close-up reveals his weathered hands gripping the steering wheel, while his eyes briefly meet the rearview mirror, hinting at countless stories shared with passengers. The scene transitions to him pausing at a red light, glancing at a photo of his family taped to the dashboard, a silent testament to his dedication and the life he supports behind the wheel."
}

def convert_to_prompt(key, description):
    """
    Converts a dictionary entry into a formatted prompt string.
    """
    
    sentences = description.split(". ")

    prompt = "prompt = (\n"
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  
            prompt += f'    "{sentence.strip()}."'
            if sentence != sentences[-1].strip():  
                prompt += "\n"
    prompt += "\n)"

    return prompt

for k, v in plain_prompts.items():
    prompt = convert_to_prompt(k, v)

    # video = pipe(prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=10, num_frames=12).frames[0]
    
    print("\n\nGenerating a synthetic image of a ", k)
    
    video = pipe(prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=20, num_frames=24).frames[0]
    
    
    export_to_video(video, f"results/{k}_Fake.mp4", fps=6)

