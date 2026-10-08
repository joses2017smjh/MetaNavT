import { z } from 'zod';
import type { SavedItem, Settings } from './contracts';
const settingsSchema = z.object({ mode: z.enum(['demo', 'connected']), metaUrl: z.string().max(2048), soccerUrl: z.string().max(2048), agentUrl: z.string().max(2048).default('') });
const savedSchema = z.array(z.object({ id: z.string(), kind: z.enum(['research', 'soccer']), title: z.string(), subtitle: z.string(), content: z.string(), savedAt: z.string(), provenance: z.string() })).max(100);
export const defaultSettings: Settings = { mode: 'demo', metaUrl: '', soccerUrl: '', agentUrl: '' };
export function loadSettings(): Settings { try { return settingsSchema.parse(JSON.parse(localStorage.getItem('agent-field.settings.v1') ?? '{}')); } catch { return defaultSettings; } }
export function loadSaved(): SavedItem[] { try { return savedSchema.parse(JSON.parse(localStorage.getItem('agent-field.saved.v1') ?? '[]')); } catch { return []; } }
export function storeSettings(settings: Settings) { localStorage.setItem('agent-field.settings.v1', JSON.stringify(settingsSchema.parse(settings))); }
export function storeSaved(saved: SavedItem[]) { localStorage.setItem('agent-field.saved.v1', JSON.stringify(savedSchema.parse(saved))); }
