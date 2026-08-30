export const OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE =
  'babel-gold-drafting:open-local-model-options';

export type OpenLocalModelOptionsMessage = {
  type: typeof OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE;
};

export function isOpenLocalModelOptionsMessage(
  value: unknown
): value is OpenLocalModelOptionsMessage {
  return (
    Boolean(value) &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    (value as { type?: unknown }).type === OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE
  );
}
