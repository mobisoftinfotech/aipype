let currentOffset = 0;
let selectedLogIndex = null;
let totalLogs = 0;
let currentPage = 1;
let totalPages = 0;
const LOGS_PER_PAGE = 50;

async function loadLogs(offset = 0) {
    try {
        document.getElementById('log-count').textContent = '(loading...)';
        document.body.classList.add('loading');
        
        const response = await fetch(`/api/logs?offset=${offset}&count=${LOGS_PER_PAGE}`);
        const data = await response.json();
        
        // Always replace logs (no append mode)
        document.getElementById('log-list').innerHTML = '';
        currentOffset = offset;
        totalLogs = data.total;
        
        if (data.logs.length === 0) {
            document.getElementById('log-list').innerHTML = '<div class="empty-state">No logs found</div>';
            updatePaginationControls();
            return;
        }
        
        const logList = document.getElementById('log-list');
        data.logs.forEach((log, index) => {
            const logElement = createLogElement(log, offset + index);
            logList.appendChild(logElement);
        });
        
        updatePaginationControls();
        
    } catch (error) {
        console.error('Error loading logs:', error);
        document.getElementById('log-count').textContent = '(error loading)';
        document.getElementById('log-list').innerHTML = '<div class="empty-state">Error loading logs</div>';
    } finally {
        document.body.classList.remove('loading');
    }
}

function createLogElement(log, index) {
    const element = document.createElement('div');
    element.className = 'log-entry';
    element.onclick = () => selectLog(log, index, element);
    
    const timestamp = new Date(log.timestamp).toLocaleString();
    
    const agentDisplay = log.agent_name ? `${escapeHtml(log.agent_name)}: ` : '';
    
    element.innerHTML = `
        <div class="log-meta">
            <div class="log-task">${agentDisplay}${escapeHtml(log.task_name)}</div>
            <div class="log-timestamp">${timestamp}</div>
        </div>
        <div class="log-provider">
            <span class="provider-tag">${escapeHtml(log.provider)}</span>
            <span class="provider-tag">${escapeHtml(log.model)}</span>
        </div>
    `;
    
    return element;
}

function selectLog(log, index, element) {
    // Remove previous selection
    document.querySelectorAll('.log-entry').forEach(el => el.classList.remove('selected'));
    
    // Add selection to current element
    element.classList.add('selected');
    selectedLogIndex = index;
    
    // Update content panel
    updateContentPanel(log);
}

function updateContentPanel(log) {
    const panelTitle = document.getElementById('panel-title');
    const panelSubtitle = document.getElementById('panel-subtitle');
    const contentAreas = document.getElementById('content-areas');
    
    const titleText = log.agent_name ? `${log.agent_name}: ${log.task_name}` : log.task_name;
    panelTitle.textContent = titleText;
    panelSubtitle.textContent = `${log.provider} • ${log.model} • ${new Date(log.timestamp).toLocaleString()}`;
    
    // Format input prompt
    let promptText = '';
    if (log.input.prompt) {
        promptText = log.input.prompt;
    } else if (log.input.context && log.input.prompt_template) {
        promptText = `Context: ${log.input.context}\n\nPrompt Template: ${log.input.prompt_template}`;
    } else {
        promptText = JSON.stringify(log.input, null, 2);
    }
    
    // Format output
    let outputText = '';
    if (log.output.content) {
        outputText = log.output.content;
    } else {
        outputText = JSON.stringify(log.output, null, 2);
    }
    
    // Add usage information if available
    if (log.output.usage) {
        const usage = log.output.usage;
        outputText += `\n\n--- Usage Information ---\n`;
        if (usage.prompt_tokens) outputText += `Prompt tokens: ${usage.prompt_tokens}\n`;
        if (usage.completion_tokens) outputText += `Completion tokens: ${usage.completion_tokens}\n`;
        if (usage.total_tokens) outputText += `Total tokens: ${usage.total_tokens}\n`;
    }
    
    contentAreas.innerHTML = `
        <div class="content-area">
            <h3>Input Prompt</h3>
            <textarea class="content-textarea" readonly>${escapeHtml(promptText)}</textarea>
        </div>
        <div class="content-area">
            <h3>LLM Output</h3>
            <textarea class="content-textarea" readonly>${escapeHtml(outputText)}</textarea>
        </div>
    `;
}

function updatePaginationControls() {
    totalPages = Math.ceil(totalLogs / LOGS_PER_PAGE);
    currentPage = Math.floor(currentOffset / LOGS_PER_PAGE) + 1;
    
    document.getElementById('prev-btn').disabled = currentPage === 1;
    document.getElementById('next-btn').disabled = currentPage === totalPages || totalPages === 0;
    document.getElementById('page-info').textContent = `${currentPage}/${totalPages || 1}`;
    
    const startRecord = totalLogs > 0 ? (currentOffset + 1) : 0;
    const endRecord = Math.min(currentOffset + LOGS_PER_PAGE, totalLogs);
    document.getElementById('log-count').textContent = `(${startRecord}-${endRecord} of ${totalLogs})`;
}

function goToNextPage() {
    if (currentPage < totalPages) {
        const newOffset = currentPage * LOGS_PER_PAGE;
        loadLogs(newOffset);
    }
}

function goToPrevPage() {
    if (currentPage > 1) {
        const newOffset = (currentPage - 2) * LOGS_PER_PAGE;
        loadLogs(newOffset);
    }
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load initial logs
loadLogs();