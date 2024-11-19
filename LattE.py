from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("maxin-cn/Latte-1")

prompt = "A weary soldier, clad in a dusty, camouflage uniform, stands solemnly in front of the camera, his eyes reflecting a deep sadness and resignation. His face, marked by the grime of battle and the weight of impending conflict, conveys a poignant awareness that the war is imminent. The background is a blur of military activity, hinting at the chaos about to unfold. His posture is rigid yet somehow defeated, as he clutches his helmet in one hand, a symbol of the protection and burden he carries. The somber lighting casts shadows over his features, emphasizing the heavy toll of his duty and the somber realization that his time to face the horrors of war has arrived."
image = pipe(prompt).images[0]