{{/*
fullnameOverride, or "<release>-<chart name or nameOverride>" (the release alone when it contains that name); at most
63 characters, since some Kubernetes name fields are DNS labels.
*/}}
{{- define "llm-mini-infer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "llm-mini-infer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "llm-mini-infer.labels" -}}
helm.sh/chart: {{ include "llm-mini-infer.chart" . }}
{{ include "llm-mini-infer.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "llm-mini-infer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llm-mini-infer.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
The pod's labels: the selector labels and .Values.podLabels. Rendering fails if they carry every label of a set in
.Values.notSelectedBy: a router that discovers its engines by label adopts any pod in its namespace whose labels match
its selector, and would send it traffic meant for its own engines.
*/}}
{{- define "llm-mini-infer.podLabels" -}}
{{- $labels := merge (dict) (include "llm-mini-infer.selectorLabels" . | fromYaml) (.Values.podLabels | default dict) }}
{{- range $set := .Values.notSelectedBy }}
{{- $matched := true }}
{{- range $k, $v := $set.labels }}
{{- if ne (toString (get $labels $k)) (toString $v) }}
{{- $matched = false }}
{{- end }}
{{- end }}
{{- if $matched }}
{{- fail (printf "llm-mini-infer's pod labels %v carry every label of %s's selector %v: it would adopt the pods; change podLabels" $labels $set.name $set.labels) }}
{{- end }}
{{- end }}
{{- toYaml $labels }}
{{- end }}

{{/*
llm-mini-infer's configuration file: .Values.config with model.path and model.revision from .Values.model.
*/}}
{{- define "llm-mini-infer.config" -}}
{{- $cfg := deepCopy (.Values.config | default dict) }}
{{- $model := deepCopy (get $cfg "model" | default dict) }}
{{- $id := required "model.id is required: a repo id, or a checkpoint's path on the cache volume" .Values.model.id }}
{{- $_ := set $model "path" $id }}
{{- if .Values.model.revision }}
{{- $_ := set $model "revision" .Values.model.revision }}
{{- end }}
{{- $_ := set $cfg "model" $model }}
{{- toYaml $cfg }}
{{- end }}

{{- define "llm-mini-infer.port" -}}
{{- (.Values.config.server | default dict).port | default 8000 }}
{{- end }}

{{- define "llm-mini-infer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "llm-mini-infer.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
