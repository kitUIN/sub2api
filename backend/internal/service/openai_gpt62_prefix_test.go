package service

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenCodeGPT62ReplayInstructions(t *testing.T) {
	const instructions = " Project rules\nsecond line "
	capture := func(t *testing.T, model, history string) gjson.Result {
		t.Helper()
		upstream := instructionsEchoUpstream(`""`)
		svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		c.Request.Header.Set("User-Agent", "opencode/latest ai-sdk")
		SetOpenAIClientTransport(c, OpenAIClientTransportHTTP)
		body := []byte(`{"model":"` + model + `","instructions":" Project rules\nsecond line ","store":false,"stream":true,"prompt_cache_key":"session","input":[` + history + `]}`)
		_, err := svc.Forward(context.Background(), c, instructionsEchoOAuthAccount(false), body)
		require.NoError(t, err)
		require.Equal(t, instructions, sseResponseInstructions(t, rec.Body.String(), "response.completed").String())
		return gjson.ParseBytes(upstream.lastBody)
	}
	const user = `{"role":"user","content":"one"}`
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		t.Run(model, func(t *testing.T) {
			first := capture(t, model, user)
			next := capture(t, model, user+`,{"role":"assistant","content":"answer"},{"role":"user","content":"two"}`)
			require.Equal(t, model, first.Get("model").String())
			require.True(t, first.Get("instructions").Exists())
			require.Equal(t, "", first.Get("instructions").String())
			require.Equal(t, "developer", first.Get("input.0.role").String())
			require.Equal(t, instructions, first.Get("input.0.content").String())
			require.Equal(t, "user", first.Get("input.1.role").String())
			require.Len(t, first.Get("input").Array(), 2)
			require.Len(t, next.Get("input").Array(), 4)
			require.Equal(t, first.Get("input").Value(), next.Get("input").Value().([]any)[:2])
			require.Equal(t, first.Get("prompt_cache_key").String(), next.Get("prompt_cache_key").String())
		})
	}
}

func TestOpenCodeGPT62ReplayKeepsTurnInstructions(t *testing.T) {
	for _, tc := range []struct {
		name, model, userAgent string
		fields                 map[string]any
	}{
		{"other client", "gpt-6.2-sol", "openai-python", nil},
		{"other model", "gpt-6.1-sol", "opencode/latest", nil},
		{"chained", "gpt-6.2-sol", "opencode/latest", map[string]any{"previous_response_id": "resp_1"}},
		{"conversation", "gpt-6.2-sol", "opencode/latest", map[string]any{"conversation": "conv_1"}},
		{"stored", "gpt-6.2-sol", "opencode/latest", map[string]any{"store": true}},
		{"no cache key", "gpt-6.2-sol", "opencode/latest", map[string]any{"prompt_cache_key": ""}},
		{"mixed guidance", "gpt-6.2-sol", "opencode/latest", map[string]any{"input": []any{map[string]any{"role": "user"}, map[string]any{"role": "developer"}}}},
		{"opaque history", "gpt-6.2-sol", "opencode/latest", map[string]any{"input": []any{map[string]any{"role": "user"}, map[string]any{"type": "item_reference", "id": "msg_1"}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := map[string]any{"instructions": "this turn", "prompt_cache_key": "session", "store": false, "input": []any{map[string]any{"role": "user"}}}
			for k, v := range tc.fields {
				body[k] = v
			}
			require.False(t, prefixOpenCodeGPT62Instructions(body, tc.model, tc.userAgent))
			require.Equal(t, "this turn", body["instructions"])
		})
	}
}

func TestGPT62OAuthInstructionsPrefixAcrossTurns(t *testing.T) {
	for _, model := range []string{"gpt-6.2-sol", "gpt-6.2-astra"} {
		t.Run(model, func(t *testing.T) {
			account := instructionsEchoOAuthAccount(false)
			account.Credentials["model_mapping"] = map[string]any{"public": model}
			capture := func(raw string) gjson.Result {
				t.Helper()
				upstream := instructionsEchoUpstream(`""`)
				_, _, err := instructionsEchoForward(t, account, upstream, raw)
				require.NoError(t, err)
				result := gjson.ParseBytes(upstream.lastBody)
				require.Equal(t, model, result.Get("model").String())
				return result
			}
			first := capture(`{"model":"public","input":[{"role":"system","content":"Answer in JSON"},{"role":"user","content":"one"}]}`)
			next := capture(`{"model":"public","text":{"format":{"type":"json_object"}},"instructions":"Only this turn","input":[{"role":"system","content":"Answer in JSON"},{"role":"user","content":"one"},{"role":"assistant","content":"{}"},{"role":"system","content":"Later guidance"},{"role":"user","content":"two"}]}`)
			require.Equal(t, "developer", first.Get("input.0.role").String())
			require.Equal(t, "Answer in JSON", first.Get("input.0.content").String())
			require.Equal(t, "", first.Get("instructions").Str)
			require.True(t, first.Get("instructions").Exists())
			require.Equal(t, "Only this turn", next.Get("instructions").String())
			require.Equal(t, first.Get("input").Value(), next.Get("input").Value().([]any)[:2])
			require.Equal(t, "Later guidance", next.Get("input.3.content").String())
			require.Equal(t, "developer", next.Get("input.3.role").String())
			clientBase := capture(`{"model":"public","instructions":"   ","input":[{"role":"developer","content":"client policy"},{"role":"user","content":"one"}]}`)
			require.Equal(t, "", clientBase.Get("instructions").String())
			require.Len(t, clientBase.Get("input").Array(), 2)
			require.Equal(t, "client policy", clientBase.Get("input.0.content").String())
			bare := capture(`{"model":"public","input":"one"}`)
			require.Equal(t, defaultCodexSynthInstructions(model), bare.Get("input.0.content").String())
			require.Equal(t, "one", bare.Get("input.1.content").String())
			require.Equal(t, "", bare.Get("instructions").String())
		})
	}
}

func TestGPT62ChainedRequestDoesNotRepeatBase(t *testing.T) {
	body := map[string]any{"model": "gpt-6.2-sol", "previous_response_id": "resp_previous", "input": "turn two"}
	require.NoError(t, applyCodexOAuthTransform(body, false, false).Error)
	require.Len(t, body["input"], 1)
	require.Equal(t, "", body["instructions"])
}

func TestGPT62WebSearchReplayKeepsToolPrefix(t *testing.T) {
	const previous = `{"model":"gpt-6.2-sol","tools":[{"type":"function","name":"work"}],"input":[{"role":"user","content":"start"},{"type":"additional_tools","role":"developer","tools":[]},{"type":"web_search_call","id":"ws_test"},{"type":"compaction_trigger"}]}`
	for _, lite := range []bool{false, true} {
		var decoded map[string]any
		require.NoError(t, json.Unmarshal([]byte(previous), &decoded))
		require.True(t, ensureOpenAIOAuthWebSearchToolForHistory(decoded, lite))
		raw, changed, err := ensureOpenAIOAuthWebSearchToolForHistoryBody([]byte(previous), lite)
		require.NoError(t, err)
		require.True(t, changed)
		expected, err := json.Marshal(decoded)
		require.NoError(t, err)
		require.JSONEq(t, string(expected), string(raw))
		result := gjson.ParseBytes(raw)
		require.Equal(t, gjson.Get(previous, "tools").Raw, result.Get("tools").Raw)
		require.Equal(t, gjson.Get(previous, "input.1").Value(), result.Get("input.1").Value())
		require.Equal(t, "web_search", result.Get("input.2.tools.0.type").String())
		require.False(t, result.Get("input.2.tools.0.external_web_access").Bool())
		require.Equal(t, "ws_test", result.Get("input.3.id").String())
		require.Equal(t, "compaction_trigger", result.Get("input.4.type").String())
		_, changed, err = ensureOpenAIOAuthWebSearchToolForHistoryBody(raw, lite)
		require.NoError(t, err)
		require.False(t, changed)
	}
}

func TestGPT62LiteNamespaceReplayKeepsPrefix(t *testing.T) {
	account := instructionsEchoOAuthAccount(false)
	account.Credentials["model_mapping"] = map[string]any{"public": "gpt-6.2-astra"}
	normalize := func(input, tools string) gjson.Result {
		t.Helper()
		body := []byte(`{"model":"public","input":` + input + `,"tools":` + tools + `}`)
		raw, _, err := normalizeOpenAIResponsesLitePayloadForAccount(body, account)
		require.NoError(t, err)
		return gjson.GetBytes(raw, "input")
	}
	const initial = `{"type":"additional_tools","role":"developer","tools":[{"type":"function","name":"native"}]},{"role":"user","content":"one"}`
	const history = `,{"role":"assistant","content":"reply"},{"role":"user","content":"two"}`
	const firstTool = `{"type":"namespace","name":"first","tools":[]}`
	const secondTool = `{"type":"namespace","name":"second","tools":[]}`
	first := normalize("["+initial+"]", "["+firstTool+"]")
	next := normalize("["+initial+history+"]", "["+firstTool+","+secondTool+"]")
	require.Len(t, first.Get("0.tools").Array(), 1)
	require.Equal(t, "first", first.Get("2.tools.0.name").String())
	require.Equal(t, first.Value(), next.Value().([]any)[:3])
	require.Equal(t, "second", next.Get("3.tools.0.name").String())
	replayed := normalize(first.Raw[:len(first.Raw)-1]+history+`,{"type":"compaction_trigger"}]`, "["+firstTool+","+secondTool+"]")
	require.Equal(t, first.Value(), replayed.Value().([]any)[:3])
	require.Equal(t, "second", replayed.Get("5.tools.0.name").String())
	require.Equal(t, "compaction_trigger", replayed.Get("6.type").String())
}

func TestGPT62ForcedTemplateKeepsLaterGuidanceInPlace(t *testing.T) {
	const raw = `{"model":"gpt-6.2-astra","input":[{"type":"message","role":"developer","content":"base"},{"role":"user","content":"one"},{"role":"developer","content":"later"}]}`
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &body))
	_, err := applyForcedCodexInstructionsTemplate(body, "operator / {{ .ExistingInstructions }}", forcedCodexInstructionsTemplateData{UpstreamModel: "gpt-6.2-astra"})
	require.NoError(t, err)
	result, err := json.Marshal(body)
	require.NoError(t, err)
	require.Equal(t, "operator / base", gjson.GetBytes(result, "input.0.content").String())
	require.Equal(t, "later", gjson.GetBytes(result, "input.2.content").String())
	require.False(t, gjson.GetBytes(result, "instructions").Exists())
}
