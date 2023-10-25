import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline

class ImageGenerator:
    def __init__(self):
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "segmind/SSD-1B", 
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                variant="fp16"
            )
            self.pipe.to("cuda")
        except RuntimeError as e:
            st.write(f"Initialization error: {e}")
            
    def text_to_image(self, prmt):
        try:
            prompt = "Design a photo of " + prmt
            neg_prompt = "ugly, blurry, poor quality"
            
            # Move model to CPU to free up GPU memory
            self.pipe.to("cpu")
            
            image = self.pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
            
            # Move model back to GPU for faster computation in future
            self.pipe.to("cuda")
            
            return image
        except RuntimeError as e:
            torch.cuda.empty_cache()  # Clear GPU cache
            return f"An error occurred during image generation: {e}"

class StreamlitApp:
    def __init__(self):
        self.generator = ImageGenerator()

    def run(self):
        st.title('ShareLabs Image AI')
        user_input = st.text_area("Enter your text here:", "Type Here")
        if st.button("Generate Image"):
            with st.spinner('Generating Image...'):
                generated_image = self.generator.text_to_image(user_input)
                if isinstance(generated_image, str):
                    st.write(generated_image)  # Display error message
                else:
                    st.image(generated_image, caption='Generated Image', use_column_width=True)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
