import gradio as gr
import torch
import os

# فحص نوع العتاد المتوفر (هل يوجد GPU أم لا؟)
IS_GPU_AVAILABLE = torch.cuda.is_available()

# تحميل النموذج فقط إذا كان هناك GPU (لتجنب انهيار المساحة المجانية)
if IS_GPU_AVAILABLE:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login
    
    # جلب مفتاح الأمان من إعدادات المساحة (Secrets)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        
    base_model_id = "NousResearch/Hermes-3-Llama-3.1-8B"
    adapter_repo_id = "maherghanem86/BlueprintFix"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token
    )

    model = PeftModel.from_pretrained(base_model, adapter_repo_id, token=hf_token)
    model.eval()

# دالة الاستدلال
def predict_bug_fix(instruction, arch_context, buggy_code):
    # إذا كانت المساحة مجانية، نعيد رداً وهمياً برمجياً لتوضيح الفكرة للجنة
    if not IS_GPU_AVAILABLE:
        mock_response = """
        **[System Notice]:** ⚠️ Running on Free CPU Tier. Real-time inference requires a GPU.
       
       
        ```
        """
        return mock_response.strip()
    
    # إذا كان الـ GPU متوفراً، نقوم بالاستدلال الحقيقي
    prompt = f"### Instruction:\n{instruction}\n\n### Architectural Context:\n{arch_context}\n\n### Buggy Code:\n{buggy_code}\n\n### Response:\n**Root Cause Analysis:**\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_output = response.split("### Response:\n")[-1].strip()
    
    if not final_output.startswith("**Root Cause"):
        final_output = "**Root Cause Analysis:**\n" + final_output
        
    return final_output

# بناء واجهة المستخدم (Gradio UI)
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🛡️ BlueprintFix

    )
    
    # عرض رسالة تحذيرية أنيقة إذا كانت المساحة مجانية
    if not IS_GPU_AVAILABLE:
        gr.Warning("⏳ You are viewing the 'Simulation Mode' on a free CPU space. The UI is fully functional, but outputs are simulated. GPU upgrade is required for real inference.")
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Parameters (المدخلات)")
            inst_input = gr.Textbox(
                label="Instruction (التعليمة)", 
                value="Analyze the bug based on the architectural context and provide the fixed code."
            )
            context_input = gr.Textbox(
                label="Architectural Context (الوثيقة المعمارية)", 
                lines=4, 
                placeholder="Paste the design rule or architectural constraint here..."
            )
            code_input = gr.Code(
                label="Buggy Code (الكود المعيب)", 
                lines=10
            )
            submit_btn = gr.Button("🚀 Analyze & Repair", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Model Response (التشخيص والإصلاح)")
            output_display = gr.Markdown(label="Root Cause & Fixed Code")

    submit_btn.click(
        fn=predict_bug_fix, 
        inputs=[inst_input, context_input, code_input], 
        outputs=output_display
    )

# إطلاق التطبيق
demo.launch()