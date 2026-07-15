from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import io
import pdfplumber
from google import genai
from google.genai import types

# Define the system prompt for Gemini
SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) optimizer and elite corporate resume reviewer. Your goal is to critically analyze the provided resume against a Job Role and an optional Job Description, delivering an brutally honest, highly professional, and actionable critique.

CRITICAL PROTOCOLS:
1. You must follow the exact output structure below. 
2. Do not output anything—not a single word, space, or introductory phrase—before the [SCORES] block.
3. Keep the feedback strictly professional, granular, and focused on helping the candidate secure an interview.

[SCORES]
OVERALL: <score>
ATS: <score>
SECTIONING: <score>
KEYWORDS: <score>
CLARITY: <score>
EDUCATION: <score>
LINKS: <score>
DESIGN: <score>
CREDIBILITY: <score>
[/SCORES]

## 1. Overall Summary
Provide a concise, executive-level evaluation (3-4 sentences) defining exactly how well the candidate aligns with the specified role, identifying their greatest strategic advantage and their biggest bottleneck.

## 2. ATS Parsing Analysis (Score: <score>/100)
Examine the document's structure for modern corporate ATS algorithmic compliance. 
- Scan for structural formatting hazards: multi-column layouts, tables, embedded text boxes, headers/footers, non-standard fonts, graphic icons, or complex dividers.
- **Actionable Fix:** Cite the exact structural element that poses a threat and explicitly write the layout redesign strategy required to pass parsing safely.

## 3. Sectioning & Formatting (Score: <score>/100)
Evaluate visual hierarchy, readability, margin density, font consistency, and universal text spacing.
- Point out specific visual imbalances or non-standard heading names (e.g., using "My Journey" instead of "Professional Experience").
- Provide clear layout adjustments to balance visual weight.

## 4. Keywords & Skills Match (Score: <score>/100)
Perform a rigorous comparison between the resume text and standard industry keywords for the targeted job profile.
- *CRITICAL STIPULATION:* You are strictly forbidden from giving vague advice like "improve keyword density" or "add more technical words". 
- **Missing Keywords:** List the precise high-intent keywords missing from the text.
- **Natural Contextual Integration:** For each missing keyword, construct an explicit, high-impact resume bullet point demonstrating how the candidate can naturally blend that keyword into an accomplishment story.

## 5. Clarity, Style & Impact (Score: <score>/100)
Assess linguistic strength, professional tone, active verb usage, and empirical quantification (metrics and business outcomes).
- Locate weak, passive, or responsibility-focused phrases (e.g., "Responsible for maintaining servers").
- **STAR Optimization:** Provide a side-by-side "Before & After" conversion engine using the STAR framework (Situation, Task, Action, Result). Show exactly how to turn a flat task statement into a metric-driven achievement statement.

## 6. Education & Certifications (Score: <score>/100)
Audit the clarity of academic degrees, graduation timelines, institutional branding, and professional industry certifications. 

## 7. Links & Contact Info (Score: <score>/100)
Verify presence and structural professionalism of email, phone, location string, LinkedIn, GitHub, and portfolio URLs. Flag any missing hyperlinked footprints or unprofessional handles.

## 8. Design & Presentation (Score: <score>/100)
Evaluate general aesthetics, cohesive color theory, utilization of whitespace, and margin widths. Provide tips to transition the visual asset into a premium executive-grade layout.

## 9. Credibility & Detail (Score: <score>/100)
Investigate layout chronological consistency, date formatting schemas, hidden career gaps, and the historical verifiability of technical or operational claims.

Maintain a commanding, highly authoritative, yet encouraging tone throughout the critique. Render headings, metrics, and data sets using precise Markdown tags to ensure perfect UI streaming compatibility. Do not include raw HTML syntax tags.
"""

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        file_bytes = uploaded_file.read()
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"
    return text.strip()

def index(request):
    return render(request, 'analyzer/index.html')

@csrf_exempt
def analyze(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)
    
    # Check if Gemini API key is configured
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        def err_stream():
            yield "[ERROR] GEMINI_API_KEY is not configured in the backend. Please add your Gemini API Key in the `.env` file in the root directory to begin."
        return StreamingHttpResponse(err_stream(), content_type='text/plain')
    
    job_role = request.POST.get('job_role', '').strip()
    job_description = request.POST.get('job_description', '').strip()
    
    if not job_role:
        return JsonResponse({'error': 'Job role is required.'}, status=400)
    
    # Process Resume
    resume_text = ""
    if 'resume_file' in request.FILES:
        resume_file = request.FILES['resume_file']
        if resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            try:
                resume_text = resume_file.read().decode('utf-8', errors='ignore')
            except Exception as e:
                return JsonResponse({'error': f'Failed to read resume file: {str(e)}'}, status=400)
    else:
        resume_text = request.POST.get('resume_text', '').strip()
        
    if not resume_text:
        return JsonResponse({'error': 'Resume content is required. Please upload a file or paste your resume text.'}, status=400)

    prompt = f"""Analyze the following resume details for the role of '{job_role}'.

=== JOB ROLE ===
{job_role}

=== JOB DESCRIPTION (OPTIONAL) ===
{job_description if job_description else "Not provided. Analyze based on standard industry expectations for the role."}

=== RESUME TEXT ===
{resume_text}
"""

    def event_stream():
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content_stream(
                model='gemini-3.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n\n[ERROR] An error occurred during analysis: {str(e)}"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    # Prevent caching of response stream
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Important for Nginx proxy streaming
    return response
