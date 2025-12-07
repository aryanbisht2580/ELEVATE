import React, { useEffect, useState } from "react";
import {
  BtnBold,
  BtnBulletList,
  BtnItalic,
  BtnLink,
  BtnNumberedList,
  BtnStrikeThrough,
  BtnUnderline,
  Editor,
  EditorProvider,
  Separator,
  Toolbar,
} from "react-simple-wysiwyg";
import { AIChatSession } from "@/Services/AiModel";
import { Button } from "../ui/button";
import { toast } from "sonner";
import { Sparkles, LoaderCircle } from "lucide-react";

function SimpeRichTextEditor({ index, onRichTextEditorChange, resumeInfo }) {
  const [value, setValue] = useState(
    resumeInfo?.projects[index]?.projectSummary || ""
  );
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    onRichTextEditorChange(value);
  }, [value]);

  const splitIntoBullets = (text) => {
    // Try to parse if it's an array format
    if (text.startsWith('[') && text.endsWith(']')) {
      try {
        const parsedArray = JSON.parse(text);
        if (Array.isArray(parsedArray)) {
          return parsedArray;
        }
      } catch (e) {
        console.error("Could not parse AI response as array:", e);
      }
    }

    // Check if the text contains dash-based bullet points (e.g., "- point")
    if (text.includes('\n- ') || text.startsWith('- ')) {
      // Split by lines that start with "- "
      const lines = text.split('\n');
      const bulletPoints = lines
        .map(line => line.trim())
        .filter(line => line.startsWith('- '))
        .map(line => line.substring(2).trim()) // Remove the "- " prefix
        .filter(s => s.length > 0);
      
      if (bulletPoints.length > 0) {
        return bulletPoints;
      }
    }
    
    // If it's plain text, try to split by common sentence separators
    // This handles responses that might have multiple sentences or points
    let sentences = text.split(/(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])/);
    
    // If the above doesn't work well, try splitting by newlines
    if (sentences.length <= 1) {
      sentences = text.split(/\n/);
    }
    
    // Clean up the sentences and remove empty strings
    return sentences
      .map(s => s.trim())
      .filter(s => s.length > 0);
  };

  const EnhanceSummaryFromAI = async () => {
    if (!value.trim()) {
      toast("Please enter some content to enhance");
      return;
    }

    setLoading(true);

    try {
      const prompt = `Please enhance and improve the following project description. Make it more professional, impactful, and keyword-rich for a resume. Format as bullet points with dashes. Keep the core meaning but make it more compelling and effective. Return only the text:\n\n${value}`;
      
      const result = await AIChatSession.sendMessage(prompt);
      const aiResponse = result.response.text();
      
      // Split the response into bullet points
      const bulletPoints = splitIntoBullets(aiResponse);
      
      // Convert to HTML list format
      const bulletHTML = `<ul>${bulletPoints.map(point => `<li>${point}</li>`).join('')}</ul>`;
      
      setValue(bulletHTML);
      onRichTextEditorChange(bulletHTML);
    } catch (error) {
      toast("Error enhancing content", "error");
      console.error("AI Enhancement Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white text-black border border-gray-300 p-4 rounded">
      <div className="flex justify-between items-center mb-2">
        <label className="text-sm font-medium">Project Description & Key Features</label>
        <Button
          onClick={EnhanceSummaryFromAI}
          disabled={loading}
          className="bg-white text-black border border-black hover:bg-gray-100 px-3 py-1 text-sm flex items-center gap-2"
        >
          {loading ? (
            <LoaderCircle className="h-4 w-4 animate-spin" />
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Enhance with AI
            </>
          )}
        </Button>
      </div>

      <EditorProvider>
        <Editor
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            onRichTextEditorChange(e.target.value);
          }}
          style={{
            backgroundColor: "white",
            color: "black",
            border: "1px solid #ccc",
            padding: "8px",
            minHeight: "150px",
          }}
        >
          <Toolbar>
            <BtnBold />
            <BtnItalic />
            <BtnUnderline />
            <BtnStrikeThrough />
            <Separator />
            <BtnNumberedList />
            <BtnBulletList />
            <Separator />
            <BtnLink />
          </Toolbar>
        </Editor>
      </EditorProvider>
    </div>
  );
}

export default SimpeRichTextEditor;
