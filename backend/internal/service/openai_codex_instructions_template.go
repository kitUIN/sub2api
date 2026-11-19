package service

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
)

type forcedCodexInstructionsTemplateData struct {
	ExistingInstructions string
	OriginalModel        string
	NormalizedModel      string
	BillingModel         string
	UpstreamModel        string
}

func applyForcedCodexInstructionsTemplate(
	reqBody map[string]any,
	templateText string,
	data forcedCodexInstructionsTemplateData,
) (bool, error) {
	if templateText == "" {
		return false, nil
	}
	prefix := openai.GPT62ModelID(data.UpstreamModel) != ""
	input, _ := reqBody["input"].([]any)
	var base map[string]any
	if prefix {
		if len(input) > 0 {
			first, _ := input[0].(map[string]any)
			if first["role"] == "developer" && first["type"] == "message" {
				if _, lossless := extractLosslessTextFromContent(first["content"]); lossless {
					base = first
					data.ExistingInstructions = strings.TrimSpace(extractTextFromContent(first["content"]))
				}
			}
		}
	} else {
		existing, _ := reqBody["instructions"].(string)
		data.ExistingInstructions = strings.TrimSpace(existing)
		if data.ExistingInstructions == "" {
			data.ExistingInstructions = extractPromptLikeInstructionsFromInput(reqBody)
		}
	}
	rendered, err := renderForcedCodexInstructionsTemplate(templateText, data)
	if err != nil {
		return false, err
	}
	if rendered == "" {
		return false, nil
	}
	if prefix {
		if base != nil {
			base["content"] = rendered
		} else {
			reqBody["input"] = append([]any{map[string]any{"type": "message", "role": "developer", "content": rendered}}, input...)
		}
		return true, nil
	}

	existing, _ := reqBody["instructions"].(string)
	if strings.TrimSpace(existing) == rendered {
		return false, nil
	}

	reqBody["instructions"] = rendered
	return true, nil
}

func renderForcedCodexInstructionsTemplate(
	templateText string,
	data forcedCodexInstructionsTemplateData,
) (string, error) {
	tmpl, err := template.New("forced_codex_instructions").Option("missingkey=zero").Parse(templateText)
	if err != nil {
		return "", fmt.Errorf("parse forced codex instructions template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("render forced codex instructions template: %w", err)
	}

	return strings.TrimSpace(buf.String()), nil
}
