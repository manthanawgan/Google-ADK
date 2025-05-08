import streamlit as st
import tempfile
import os
import subprocess
import re
import base64
from pathlib import Path
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in env variables")


# App Configuration
APP_NAME = "manim_animation_agent"
MODEL_ID = "gemini-1.5-pro"

# Animation Templates Library
ANIMATION_TEMPLATES = {
    "geometric": {
        "circle": """
from manim import *

class CircleAnimation(Scene):
    def construct(self):
        circle = Circle()
        self.play(ShowCreation(circle))
        self.play(circle.animate.scale(2))
        self.wait(1)
        """,
        "square": """ 
from manim import *

class SquareAnimation(Scene):
    def construct(self):
        square = Square()
        self.play(ShowCreation(square))
        self.play(square.animate.rotate(PI/4))
        self.wait(1)
        """
    },
    "mathematical": {
        "pythagorean": """
from manim import *

class PythagoreanTheorem(Scene):
    def construct(self):
        # create a right triangle
        triangle = Polygon(
            ORIGIN, RIGHT*3, UP*4, 
            color=WHITE
        )
        
        # create squares on each side
        square_a = Square(side_length=3).next_to(triangle, DOWN, buff=0.5)
        square_b = Square(side_length=4).next_to(triangle, RIGHT, buff=0.5)
        square_c = Square(side_length=5).next_to(triangle.get_center(), UP+RIGHT, buff=0.5)
        
        # Labels
        label_a = MathTex("a^2").move_to(square_a.get_center())
        label_b = MathTex("b^2").move_to(square_b.get_center())
        label_c = MathTex("c^2").move_to(square_c.get_center())
        equation = MathTex("a^2 + b^2 = c^2").to_edge(DOWN)
        
        # Animations
        self.play(ShowCreation(triangle))
        self.play(ShowCreation(square_a), ShowCreation(square_b), ShowCreation(square_c))
        self.play(Write(label_a), Write(label_b), Write(label_c))
        self.play(Write(equation))
        self.wait(2)
        """
    },
    "text": {
        "title": """
from manim import *

class TitleAnimation(Scene):
    def construct(self):
        title = Text("{text}")
        self.play(Write(title))
        self.play(title.animate.scale(1.5))
        self.wait(1)
        """
    },
    "function": {
        "graph": """
from manim import *

class FunctionGraph(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-5, 5, 1],
            axis_config={"include_tip": False}
        )
        
        # create the graph
        graph = axes.plot(lambda x: {function}, color=BLUE)
        graph_label = MathTex(r"{function_tex}").next_to(graph, UP)
        
        # Animations
        self.play(ShowCreation(axes))
        self.play(ShowCreation(graph))
        self.play(Write(graph_label))
        self.wait(1)
        """
    }
}

# Generate Manim Code
def generate_manim_code(prompt: str) -> dict:
    """Generates Python code using the Manim library based on the user's prompt."""
    prompt_lower = prompt.lower()
    
    # Check for specific animation types
    if "circle" in prompt_lower:
        code = ANIMATION_TEMPLATES["geometric"]["circle"]
    elif "square" in prompt_lower:
        code = ANIMATION_TEMPLATES["geometric"]["square"]
    elif any(term in prompt_lower for term in ["pythagorean", "pythagoras", "a^2+b^2=c^2"]):
        code = ANIMATION_TEMPLATES["mathematical"]["pythagorean"]
    elif "title" in prompt_lower or "text" in prompt_lower:
        # Extract the text content from the prompt
        text_match = re.search(r'text[:\s]+["\']*([^"\']+)', prompt_lower)
        text = text_match.group(1) if text_match else "Sample Title"
        code = ANIMATION_TEMPLATES["text"]["title"].format(text=text)
    elif "function" in prompt_lower or "graph" in prompt_lower:
        # Extract function expression
        function_match = re.search(r'function[:\s]+([^,]+)', prompt_lower)
        function = function_match.group(1) if function_match else "x**2"
        # Clean up function for display
        function_tex = function.replace("**", "^").replace("*", "\\cdot ")
        code = ANIMATION_TEMPLATES["function"]["graph"].format(
            function=function, 
            function_tex=function_tex
        )
    else:
        # Default to circle if nothing matches
        code = ANIMATION_TEMPLATES["geometric"]["circle"]
    
    return {"status": "success", "code": code.strip()}

# Execute and Render Manim Code
def execute_manim_code(code: str) -> dict:
    """Executes the generated Manim code and returns path to rendered video."""
    # Create temporary file with the generated code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
    
    # Extract class name for Manim execution
    class_match = re.search(r'class\s+(\w+)\(Scene\)', code)
    if not class_match:
        return {"status": "error", "message": "Could not find Scene class in code"}
    
    scene_class = class_match.group(1)
    
    # Create directory for output
    output_dir = tempfile.mkdtemp()
    
    try:
        # Execute Manim command to render the animation
        cmd = [
            "manim", 
            temp_file_path, 
            scene_class,
            "-o", "animation_output",
            "--media_dir", output_dir,
            "--format", "mp4"  # Ensure mp4 format
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"status": "error", "message": f"Manim execution failed: {result.stderr}"}
        
        # Find generated video file
        media_path = Path(output_dir) / "videos" / Path(temp_file_path).stem / "1080p60" / "animation_output.mp4"
        
        if not media_path.exists():
            return {"status": "error", "message": "Could not find rendered animation"}
            
        return {"status": "success", "video_path": str(media_path)}
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

# Analyze Code Complexity
def analyze_complexity(code: str) -> dict:
    """Estimates the complexity of Manim code based on features."""
    lines = code.count('\n')
    has_animations = any(anim in code for anim in ["Create", "Transform", "FadeIn", "Animate", "play"])
    has_classes = "class" in code
    has_nested_calls = code.count("(") > 10
    
    score = 0
    score += lines / 10
    score += 1 if has_animations else 0
    score += 1 if has_classes else 0
    score += 1 if has_nested_calls else 0
    
    if score > 5:
        difficulty = "hard"
    elif score > 3:
        difficulty = "medium"
    else:
        difficulty = "easy"
    
    return {"difficulty": difficulty, "score": round(score, 2)}

# Combined Function for Streamlit
def generate_and_render_animation(prompt: str) -> dict:
    """Generates and renders a Manim animation based on the user's prompt."""
    # Step 1: Generate code
    code_result = generate_manim_code(prompt)
    
    if code_result["status"] != "success":
        return {"status": "error", "message": "Failed to generate code"}
    
    # Step 2: Analyze complexity
    complexity = analyze_complexity(code_result["code"])
    
    # Step 3: Execute the code and render
    execution_result = execute_manim_code(code_result["code"])
    
    if execution_result["status"] != "success":
        return {
            "status": "error", 
            "message": execution_result["message"],
            "code": code_result["code"],
            "complexity": complexity
        }
    
    # Return all information
    return {
        "status": "success", 
        "code": code_result["code"],
        "video_path": execution_result["video_path"],
        "complexity": complexity
    }

# Create Google ADK Agent
def setup_adk_agent():
    animation_tool = FunctionTool(func=generate_and_render_animation)
    
    agent = Agent(
        model=MODEL_ID,
        name="manim_animation_agent",
        instructions="""
        You are an AI Agent that generates Manim Python code from user prompts and renders animations.
        First, use the `generate_and_render_animation` tool to generate the code and render the animation.
        Respond with the code and explain what the animation shows.
        """,
        tools=[animation_tool]
    )
    
    return agent

# Helper function to get video as base64
def get_video_base64(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        encoded_video = base64.b64encode(video_bytes).decode()
        return encoded_video

# Streamlit App
def main():
    st.set_page_config(page_title="Manim Animation Generator", page_icon="🎬", layout="wide")
    
    st.title("🎬 AI-Powered Manim Animation Generator")
    st.markdown("""
    Describe the animation you want to create, and our AI will generate it using the Manim library!
    Try prompts like:
    - "Create a growing circle"
    - "Show the Pythagorean theorem"
    - "Create a title with text: Hello World"
    - "Graph the function x^2 + 2x - 3"
    """)
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # User input
    with st.form("animation_form"):
        user_prompt = st.text_input("Describe your animation:")
        submit_button = st.form_submit_button("Generate Animation")
    
    # Process form submission
    if submit_button and user_prompt:
        with st.spinner("Generating animation..."):
            # Direct approach - no need for full ADK pipeline in Streamlit
            result = generate_and_render_animation(user_prompt)
            
            if result["status"] == "success":
                # Display result
                st.success("Animation generated successfully!")
                
                # Add to history
                st.session_state.history.append({
                    "prompt": user_prompt,
                    "code": result["code"],
                    "video_path": result["video_path"],
                    "complexity": result["complexity"]
                })
                
                # Display the most recent result
                latest = st.session_state.history[-1]
                
                # Display animation
                st.subheader("Your Animation")
                encoded_video = get_video_base64(latest["video_path"])
                st.markdown(
                    f"""
                    <video controls autoplay>
                        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                    </video>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display code
                st.subheader("Generated Manim Code")
                st.code(latest["code"], language="python")
                
                # Display complexity
                st.subheader("Animation Complexity")
                st.write(f"Difficulty: {latest['complexity']['difficulty']}")
                st.write(f"Complexity Score: {latest['complexity']['score']}")
                
                # Download options
                st.download_button(
                    label="Download Animation",
                    data=open(latest["video_path"], "rb").read(),
                    file_name="manim_animation.mp4",
                    mime="video/mp4"
                )
                
                # Create Python file for download
                st.download_button(
                    label="Download Python Code",
                    data=latest["code"],
                    file_name="manim_animation.py",
                    mime="text/plain"
                )
            else:
                st.error(f"Error: {result['message']}")
                if "code" in result:
                    st.subheader("Generated Code (with error)")
                    st.code(result["code"], language="python")
    
    # Show history
    if st.session_state.history and len(st.session_state.history) > 1:
        st.subheader("History")
        for i, item in enumerate(st.session_state.history[:-1]):  # Skip the latest which is already shown
            with st.expander(f"Animation {i+1}: {item['prompt']}"):
                st.code(item["code"], language="python")
                st.write(f"Complexity: {item['complexity']['difficulty']} ({item['complexity']['score']})")
                
                # Show video for this historical item
                encoded_video = get_video_base64(item["video_path"])
                st.markdown(
                    f"""
                    <video width="100%" controls>
                        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                    </video>
                    """, 
                    unsafe_allow_html=True
                )

# Deployment instructions
def show_deployment_instructions():
    st.sidebar.title("Deployment Instructions")
    st.sidebar.markdown("""
    ### How to deploy this app:
    
    1. **Install required packages**:
       ```
       pip install streamlit google-adk manim
       ```
       
    2. **Set environment variables**:
       ```
       export GOOGLE_API_KEY=your_api_key_here
       ```
       
    3. **Deploy to Streamlit Cloud**:
       - Connect your GitHub repo to Streamlit Cloud
       - Select this file as the main file
       - Add your API key as a secret
       
    4. **Alternative: Run on a server**:
       ```
       streamlit run app.py --server.port 8501
       ```
       
    5. **Requirements**:
       - Server needs Manim and its dependencies installed
       - FFmpeg for video processing
       - Cairo for rendering
    """)

# Main execution
if __name__ == "__main__":
    # Add deployment instructions to sidebar
    show_deployment_instructions()
    
    # Run the main application
    main()