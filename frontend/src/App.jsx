import React, { useCallback, useEffect, useRef, useState } from 'react';

const MODEL_API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2';
const STORAGE_KEY = 'gemini-chat-app-data';
const TOKEN_STORAGE_KEY = 'gemini-chat-hf-token';

const BASE_FILESYSTEM = {
  '/': {
    type: 'dir',
    children: {
      home: {
        type: 'dir',
        children: {
          user: {
            type: 'dir',
            children: {
              'welcome.txt': {
                type: 'file',
                content: 'Welcome to the virtual terminal! Type `help` for a list of commands.'
              }
            }
          }
        }
      },
      bin: { type: 'dir', children: {} },
      etc: { type: 'dir', children: {} }
    }
  }
};

const deepClone = (value) => JSON.parse(JSON.stringify(value));

const createFilesystemSnapshot = () => deepClone(BASE_FILESYSTEM);

const cloneFilesystem = (filesystem) => {
  try {
    if (!filesystem || typeof filesystem !== 'object' || !filesystem['/']) {
      return createFilesystemSnapshot();
    }
    return deepClone(filesystem);
  } catch (error) {
    console.warn('Failed to clone filesystem, restoring initial state.', error);
    return createFilesystemSnapshot();
  }
};

const sanitizeMessages = (messages) => {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages
    .filter((message) => message && typeof message.content === 'string')
    .map((message) => {
      const role = message.role === 'assistant' ? 'assistant' : 'user';
      const sanitizedMessage = {
        role,
        content: message.content
      };
      if (message.type === 'command' || message.type === 'output') {
        sanitizedMessage.type = message.type;
      }
      if (message.executedCommand && typeof message.executedCommand === 'string') {
        sanitizedMessage.executedCommand = message.executedCommand;
      }
      return sanitizedMessage;
    });
};

const sanitizeChat = (chat) => {
  if (!chat || typeof chat !== 'object') {
    return {
      messages: [],
      agentMode: 'chat',
      filesystem: createFilesystemSnapshot(),
      cwd: '/home/user'
    };
  }

  return {
    messages: sanitizeMessages(chat.messages),
    agentMode: chat.agentMode === 'terminal' ? 'terminal' : 'chat',
    filesystem: cloneFilesystem(chat.filesystem),
    cwd: typeof chat.cwd === 'string' ? chat.cwd : '/home/user'
  };
};

const resolvePath = (path, cwd) => {
  if (!path || path === '.') {
    return cwd;
  }
  if (path.startsWith('/')) {
    return path === '/' ? '/' : `/${path.split('/').filter(Boolean).join('/')}`;
  }
  const parts = [...cwd.split('/'), ...path.split('/')].filter(Boolean);
  const resolved = [];
  for (const part of parts) {
    if (part === '..') {
      resolved.pop();
    } else if (part !== '.') {
      resolved.push(part);
    }
  }
  return `/${resolved.join('/')}` || '/';
};

const getNode = (path, filesystem) => {
  const segments = path.split('/').filter(Boolean);
  let node = filesystem['/'];
  for (const segment of segments) {
    if (!node || node.type !== 'dir' || !node.children[segment]) {
      return null;
    }
    node = node.children[segment];
  }
  return node;
};

const createNode = (path, filesystem, type, content) => {
  const segments = path.split('/').filter(Boolean);
  if (segments.length === 0) {
    return { success: false, error: 'cannot create root directory' };
  }
  const name = segments.pop();
  let parent = filesystem['/'];
  const traversed = [];

  for (const segment of segments) {
    traversed.push(segment);
    if (!parent || parent.type !== 'dir') {
      return { success: false, error: `Path is not a directory: /${traversed.join('/')}` };
    }
    if (!parent.children[segment]) {
      return { success: false, error: `Directory not found: /${traversed.join('/')}` };
    }
    parent = parent.children[segment];
  }

  if (!parent || parent.type !== 'dir') {
    return { success: false, error: 'Path is not a directory' };
  }

  const existing = parent.children[name];
  if (existing) {
    if (existing.type !== type) {
      return { success: false, error: `${name} already exists and is not a ${type}` };
    }
    if (type === 'file' && typeof content === 'string') {
      existing.content = content;
    }
    return { success: true, node: existing };
  }

  parent.children[name] =
    type === 'dir'
      ? { type: 'dir', children: {} }
      : { type: 'file', content: typeof content === 'string' ? content : '' };

  return { success: true, node: parent.children[name] };
};

const stripWrappingQuotes = (value) => {
  if (!value) {
    return '';
  }
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  return value;
};

const executeCommand = (command, filesystem, cwd) => {
  const trimmed = command.trim();
  if (!trimmed) {
    return { output: '', fs: filesystem, cwd };
  }

  const [cmd, ...args] = trimmed.split(/\s+/);
  let output = '';

  switch (cmd) {
    case 'help':
      output = [
        'Available commands:',
        'ls, cd, pwd, cat, echo, mkdir, touch, help',
        '',
        'You can also ask the AI to generate commands for you, e.g., "list all files in the current directory".'
      ].join('\n');
      break;
    case 'ls': {
      const targetPath = args[0] ? resolvePath(args[0], cwd) : cwd;
      const node = getNode(targetPath, filesystem);
      if (node && node.type === 'dir') {
        output = Object.keys(node.children).sort().join('\n');
      } else if (node && node.type === 'file') {
        output = args[0] || '.';
      } else {
        output = `ls: cannot access '${args[0] || '.'}': No such file or directory`;
      }
      break;
    }
    case 'pwd':
      output = cwd;
      break;
    case 'cd': {
      const target = args[0] ? resolvePath(args[0], cwd) : '/home/user';
      const node = getNode(target, filesystem);
      if (node && node.type === 'dir') {
        return { output: '', fs: filesystem, cwd: target };
      }
      output = `cd: no such file or directory: ${args[0] || target}`;
      break;
    }
    case 'cat': {
      if (!args[0]) {
        output = 'cat: missing operand';
        break;
      }
      const target = resolvePath(args[0], cwd);
      const node = getNode(target, filesystem);
      if (node && node.type === 'file') {
        output = node.content;
      } else {
        output = `cat: ${args[0]}: No such file or directory`;
      }
      break;
    }
    case 'mkdir': {
      if (!args[0]) {
        output = 'mkdir: missing operand';
        break;
      }
      const target = resolvePath(args[0], cwd);
      const result = createNode(target, filesystem, 'dir');
      if (!result.success) {
        output = `mkdir: ${result.error}`;
      }
      break;
    }
    case 'touch': {
      if (!args[0]) {
        output = 'touch: missing operand';
        break;
      }
      const target = resolvePath(args[0], cwd);
      const result = createNode(target, filesystem, 'file');
      if (!result.success) {
        output = `touch: ${result.error}`;
      }
      break;
    }
    case 'echo': {
      const echoContent = command.slice(command.indexOf('echo') + 4).trim();
      if (!echoContent) {
        output = '';
        break;
      }
      const match = echoContent.match(/^(.*?)(?:\s*(>>?)\s*(\S+))?$/s);
      if (!match) {
        output = stripWrappingQuotes(echoContent);
        break;
      }
      const [, body, redirect, target] = match;
      const text = stripWrappingQuotes(body.trim());
      if (!redirect || !target) {
        output = text;
        break;
      }
      const resolvedTarget = resolvePath(target.trim(), cwd);
      let node = getNode(resolvedTarget, filesystem);
      if (!node) {
        const result = createNode(resolvedTarget, filesystem, 'file', redirect === '>>' ? '' : text);
        if (!result.success) {
          output = `echo: ${result.error}`;
          break;
        }
        node = getNode(resolvedTarget, filesystem);
      }
      if (!node || node.type !== 'file') {
        output = `echo: cannot write to '${target}': Is a directory`;
        break;
      }
      if (redirect === '>>') {
        node.content = node.content ? `${node.content}\n${text}` : text;
      } else {
        node.content = text;
      }
      break;
    }
    default:
      output = `${cmd}: command not found`;
  }
  return { output, fs: filesystem, cwd };
};

const buildPromptFromMessages = (messages) =>
  messages
    .filter((message) => !message.type && (message.role === 'user' || message.role === 'assistant'))
    .map((message) => (message.role === 'user' ? `[INST] ${message.content} [/INST]` : message.content))
    .join('\n');

const SendIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-white">
    <path d="M10 14L21 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M21 3L14.5 21L10 14L3 9.5L21 3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const NewChatIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 5V19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const CopyIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M5 15H4C2.89543 15 2 14.1046 2 13V4C2 2.89543 2.89543 2 4 2H13C14.1046 2 15 2.89543 15 4V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const UserIcon = () => (
  <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-white font-bold text-sm shrink-0">
    You
  </div>
);

const ModelIcon = () => (
  <div className="w-8 h-8 rounded-full bg-teal-500 flex items-center justify-center shrink-0">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2 17L12 22L22 17" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2 12L12 17L22 12" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </div>
);

const TerminalIcon = () => (
  <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center shrink-0">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-green-400">
      <path d="M4 17L10 11L4 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 19H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </div>
);

const SimpleMarkdownRenderer = ({ content }) => {
  if (!content) return null;
  let html = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\n/g, '<br />');
  // eslint-disable-next-line react/no-danger
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
};

const MarkdownContent = ({ content }) => {
  if (!content) {
    return null;
  }
  const parts = content.split(/(```[\s\S]*?```)/g);
  return (
    <div className="prose prose-invert prose-sm max-w-none">
      {parts.map((part, index) => {
        const match = part.match(/```(\w+)?\n([\s\S]+)```/);
        if (match) {
          const language = match[1] || 'text';
          const code = match[2];
          return (
            <div key={index} className="not-prose relative my-4 rounded-md bg-gray-900">
              <div className="flex items-center justify-between px-4 py-2 bg-gray-800 rounded-t-md">
                <span className="text-xs text-gray-400">{language}</span>
                <button
                  type="button"
                  onClick={() => navigator.clipboard.writeText(code)}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
                >
                  <CopyIcon /> Copy
                </button>
              </div>
              <pre className="p-4 overflow-x-auto text-sm">
                <code>{code}</code>
              </pre>
            </div>
          );
        }
        return <SimpleMarkdownRenderer key={index} content={part} />;
      })}
    </div>
  );
};

function GeminiChat() {
  const [chats, setChats] = useState({});
  const [currentChatId, setCurrentChatId] = useState(null);
  const [agentMode, setAgentMode] = useState('chat');
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hfToken, setHfToken] = useState(() => {
    if (typeof window !== 'undefined') {
      const savedToken = window.localStorage.getItem(TOKEN_STORAGE_KEY);
      if (savedToken) {
        return savedToken;
      }
    }
    return import.meta.env?.VITE_HF_API_TOKEN || '';
  });
  const chatEndRef = useRef(null);

  const handleNewChat = useCallback(() => {
    const newChatId = `chat_${Date.now()}`;
    const newChat = {
      messages: [],
      agentMode: 'chat',
      filesystem: createFilesystemSnapshot(),
      cwd: '/home/user'
    };
    setChats((previousChats) => ({ ...previousChats, [newChatId]: newChat }));
    setCurrentChatId(newChatId);
    setAgentMode('chat');
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const savedDataRaw = window.localStorage.getItem(STORAGE_KEY);
    if (!savedDataRaw) {
      handleNewChat();
      return;
    }
    try {
      const savedData = JSON.parse(savedDataRaw);
      if (!savedData || typeof savedData !== 'object' || typeof savedData.chats !== 'object') {
        throw new Error('Invalid data format');
      }
      const restoredChats = Object.entries(savedData.chats).reduce((accumulator, [chatId, chatValue]) => {
        accumulator[chatId] = sanitizeChat(chatValue);
        return accumulator;
      }, {});
      const chatIds = Object.keys(restoredChats);
      if (chatIds.length === 0) {
        handleNewChat();
        return;
      }
      setChats(restoredChats);
      const savedChatId = savedData.currentChatId;
      const activeChatId = savedChatId && restoredChats[savedChatId] ? savedChatId : chatIds[0];
      setCurrentChatId(activeChatId);
      setAgentMode(restoredChats[activeChatId].agentMode);
    } catch (error) {
      console.error('Failed to restore saved chats:', error);
      handleNewChat();
    }
  }, [handleNewChat]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    if (Object.keys(chats).length === 0 || !currentChatId) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const payload = {
      chats,
      currentChatId
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }, [chats, currentChatId]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    if (!hfToken) {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    } else {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, hfToken);
    }
  }, [hfToken]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chats, currentChatId, isLoading]);

  const currentChat = currentChatId ? chats[currentChatId] : null;
  const currentMessages = currentChat?.messages ?? [];

  const handleSendMessage = useCallback(
    async (event) => {
      event.preventDefault();
      if (!currentChatId || !inputValue.trim() || isLoading) {
        return;
      }
      const currentChatSnapshot = chats[currentChatId];
      if (!currentChatSnapshot) {
        return;
      }

      const userInput = inputValue.trim();
      setInputValue('');
      setIsLoading(true);

      if (currentChatSnapshot.agentMode === 'chat') {
        const userMessage = { role: 'user', content: userInput };
        setChats((previous) => ({
          ...previous,
          [currentChatId]: {
            ...currentChatSnapshot,
            messages: [...currentChatSnapshot.messages, userMessage]
          }
        }));

        if (!hfToken) {
          const warningMessage = {
            role: 'assistant',
            content:
              'Please provide a Hugging Face API token to enable responses. You can add it from the sidebar settings.'
          };
          setChats((previous) => {
            const chat = previous[currentChatId];
            if (!chat) return previous;
            return {
              ...previous,
              [currentChatId]: {
                ...chat,
                messages: [...chat.messages, warningMessage]
              }
            };
          });
          setIsLoading(false);
          return;
        }

        const prompt = buildPromptFromMessages([...currentChatSnapshot.messages, userMessage]);
        try {
          const response = await fetch(MODEL_API_URL, {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${hfToken}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              inputs: prompt,
              parameters: { return_full_text: false, max_new_tokens: 512 }
            })
          });

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error: ${response.status} ${errorText}`);
          }

          const results = await response.json();
          const generated = Array.isArray(results)
            ? results[0]?.generated_text
            : results?.generated_text || results?.content;
          const modelResponse = generated?.trim() || "Sorry, I couldn't generate a response.";

          setChats((previous) => {
            const chat = previous[currentChatId];
            if (!chat) {
              return previous;
            }
            return {
              ...previous,
              [currentChatId]: {
                ...chat,
                messages: [...chat.messages, { role: 'assistant', content: modelResponse }]
              }
            };
          });
        } catch (error) {
          console.error('Error fetching from API:', error);
          setChats((previous) => {
            const chat = previous[currentChatId];
            if (!chat) {
              return previous;
            }
            return {
              ...previous,
              [currentChatId]: {
                ...chat,
                messages: [
                  ...chat.messages,
                  { role: 'assistant', content: `Error: ${error.message}` }
                ]
              }
            };
          });
        } finally {
          setIsLoading(false);
        }
      } else {
        const commandMessage = { role: 'user', content: userInput, type: 'command' };
        setChats((previous) => ({
          ...previous,
          [currentChatId]: {
            ...currentChatSnapshot,
            messages: [...currentChatSnapshot.messages, commandMessage]
          }
        }));

        let commandToExecute = userInput;
        const basicCommands = ['ls', 'cd', 'pwd', 'cat', 'mkdir', 'touch', 'echo', 'help'];
        const isNaturalLanguage = !basicCommands.includes(userInput.trim().split(' ')[0]);
        const currentFilesystem = cloneFilesystem(currentChatSnapshot.filesystem);
        const currentCwd = currentChatSnapshot.cwd || '/home/user';

        if (isNaturalLanguage) {
          if (!hfToken) {
            commandToExecute = 'echo "AI command generation requires a Hugging Face API token."';
          } else {
            try {
              const prompt = `You are a Linux terminal assistant. Based on the user's request, provide ONLY the single, most appropriate bash command to achieve it. Do not provide any explanation or surrounding text. User request: "${userInput}"`;
              const response = await fetch(MODEL_API_URL, {
                method: 'POST',
                headers: {
                  Authorization: `Bearer ${hfToken}`,
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({ inputs: prompt, parameters: { return_full_text: false, max_new_tokens: 50 } })
              });

              if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error: ${response.status} ${errorText}`);
              }

              const results = await response.json();
              const generatedCommand = Array.isArray(results)
                ? results[0]?.generated_text
                : results?.generated_text || '';
              commandToExecute = (generatedCommand || '').trim() || "echo 'AI could not generate a command.'";
            } catch (error) {
              console.error('Error generating command:', error);
              commandToExecute = `echo "Error getting AI command: ${error.message}"`;
            }
          }
        }

        const { output, fs: newFilesystem, cwd: newCwd } = executeCommand(commandToExecute, currentFilesystem, currentCwd);
        const outputMessage = {
          role: 'assistant',
          content: output,
          type: 'output',
          executedCommand: isNaturalLanguage ? commandToExecute : null
        };

        setChats((previous) => {
          const chat = previous[currentChatId];
          if (!chat) {
            return previous;
          }
          return {
            ...previous,
            [currentChatId]: {
              ...chat,
              messages: [...chat.messages, outputMessage],
              filesystem: newFilesystem,
              cwd: newCwd || chat.cwd
            }
          };
        });
        setIsLoading(false);
      }
    },
    [agentMode, chats, currentChatId, hfToken, inputValue, isLoading]
  );

  const updateAgentMode = (mode) => {
    if (!currentChatId) {
      return;
    }
    setAgentMode(mode);
    setChats((previous) => {
      const chat = previous[currentChatId];
      if (!chat) {
        return previous;
      }
      return {
        ...previous,
        [currentChatId]: {
          ...chat,
          agentMode: mode
        }
      };
    });
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      <div className="w-72 bg-gray-950 p-4 flex flex-col border-r border-gray-800">
        <button
          type="button"
          onClick={handleNewChat}
          className="flex items-center justify-center gap-2 w-full px-4 py-2 mb-4 text-sm font-medium bg-indigo-600 hover:bg-indigo-500 rounded-md transition-colors"
        >
          <NewChatIcon /> New Chat
        </button>
        <div className="flex-1 overflow-y-auto">
          <span className="text-xs text-gray-500 font-semibold uppercase">History</span>
          <ul className="mt-2 space-y-1">
            {Object.keys(chats)
              .sort((a, b) => (a < b ? 1 : -1))
              .map((chatId) => (
                <li key={chatId}>
                  <button
                    type="button"
                    onClick={() => {
                      setCurrentChatId(chatId);
                      setAgentMode(chats[chatId].agentMode);
                    }}
                    className={`w-full text-left px-3 py-2 text-sm truncate rounded-md ${
                      currentChatId === chatId ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                    }`}
                  >
                    {chats[chatId].messages[0]?.content || `New ${chats[chatId].agentMode} session`}
                  </button>
                </li>
              ))}
          </ul>
        </div>
        <div className="pt-4 border-t border-gray-800">
          <label className="block text-xs text-gray-500 font-semibold uppercase mb-2" htmlFor="hf-token">
            Hugging Face API Token
          </label>
          <input
            id="hf-token"
            type="password"
            placeholder="hf_xxx..."
            value={hfToken}
            onChange={(event) => setHfToken(event.target.value.trim())}
            className="w-full rounded-md bg-gray-800 border border-gray-700 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <p className="text-[11px] text-gray-500 mt-2 leading-snug">
            Token is stored locally in your browser. Leave blank to disable API requests.
          </p>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {currentChat && (
          <div className="border-b border-gray-800">
            <div className="max-w-3xl mx-auto px-6 flex">
              <button
                type="button"
                onClick={() => updateAgentMode('chat')}
                className={`py-3 px-4 text-sm font-medium ${
                  agentMode === 'chat' ? 'border-b-2 border-indigo-500 text-white' : 'text-gray-400'
                }`}
              >
                Chat
              </button>
              <button
                type="button"
                onClick={() => updateAgentMode('terminal')}
                className={`py-3 px-4 text-sm font-medium ${
                  agentMode === 'terminal' ? 'border-b-2 border-indigo-500 text-white' : 'text-gray-400'
                }`}
              >
                Terminal
              </button>
            </div>
          </div>
        )}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-3xl mx-auto">
            {currentMessages.length === 0 && !isLoading && (
              <div className="text-center text-gray-500 mt-20">
                <h1 className="text-3xl font-bold text-gray-300">
                  Gemini {agentMode === 'chat' ? 'Chat' : 'Terminal'}
                </h1>
                <p className="mt-2">
                  {agentMode === 'chat'
                    ? 'Powered by open source models via Hugging Face Inference.'
                    : 'Virtual Linux environment with AI assistance.'}
                </p>
              </div>
            )}
            {currentMessages.map((message, index) => {
              if (agentMode === 'terminal') {
                return (
                  <div key={index}>
                    {message.type === 'command' && (
                      <div className="flex items-center gap-4 my-2 font-mono text-sm">
                        <div className="flex-shrink-0 text-green-400">
                          <span className="text-indigo-400">user@gemini</span>:
                          <span className="text-blue-400">{currentChat.cwd}</span>$
                        </div>
                        <div className="flex-grow break-words">{message.content}</div>
                      </div>
                    )}
                    {message.type === 'output' && (
                      <div className="mb-4">
                        {message.executedCommand && (
                          <p className="text-xs text-yellow-500 my-2 font-mono">
                            AI generated command:{' '}
                            <code className="bg-gray-800 px-1 py-0.5 rounded">{message.executedCommand}</code>
                          </p>
                        )}
                        <pre className="text-sm whitespace-pre-wrap bg-gray-800/50 rounded-md p-3 border border-gray-700">
                          {message.content}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              }
              return (
                <div key={index} className="flex items-start gap-4 my-6">
                  {message.role === 'user' ? <UserIcon /> : <ModelIcon />}
                  <div className={`w-full ${message.role === 'user' ? '' : 'bg-gray-800/50 p-4 rounded-lg'}`}>
                    <MarkdownContent content={message.content} />
                  </div>
                </div>
              );
            })}
            {isLoading && (
              <div className="flex items-start gap-4 my-6">
                {agentMode === 'chat' ? <ModelIcon /> : <TerminalIcon />}
                <div className="w-full bg-gray-800/50 p-4 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse [animation-delay:0.4s]" />
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </main>
        <footer className="p-6 border-t border-gray-800">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative">
              <textarea
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSendMessage(event);
                  }
                }}
                placeholder={agentMode === 'chat' ? 'Message Gemini Chat...' : 'Enter a command or ask the AI...'}
                rows={1}
                className={`w-full bg-gray-800 border border-gray-700 rounded-lg py-3 pr-14 pl-4 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  agentMode === 'terminal' ? 'font-mono' : ''
                }`}
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-indigo-600 rounded-full disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-indigo-500 transition-colors"
              >
                <SendIcon />
              </button>
            </form>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default GeminiChat;
