() => {
    if (!document.cookie.includes('session=')) {
        if (CONSENT_POPUP) window.alert("By using this app, you agree that submitted data is logged (only when clicking the feedback buttons) and may be used for research purposes.\n\nPlease refrain from submitting any personal or confidential information.");
        const date = new Date(+new Date() + 10 * 365 * 24 * 60 * 60 * 1000);
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        document.cookie = `session=${Array(32).fill().map(() => chars.charAt(Math.floor(Math.random() * chars.length))).join('')}; expires=${date.toUTCString()}; path=/`;
    }

    // Word count for textareas
    const wordCountHandler = (event) => {
        const words = event.target.value.split(/\s+/).filter(Boolean).length ?? 0;
        const label = event.target.previousElementSibling;
        label.style = words ? `--word-count: " (${words} words)"` : '';
    }
    const wordCountFields = document.querySelectorAll('.word-count textarea');
    wordCountFields.forEach((t) => t.addEventListener('input', wordCountHandler));

    // Add listener to example prompt buttons to set wordcount on selection
    document.querySelectorAll('.gallery, #clear')?.forEach((item) => {
        item.addEventListener('click', () => setTimeout(() => {
            wordCountFields.forEach((t) => wordCountHandler({ target: t }))
        }, 300));
    });

    // Compressed diff
    const diff = document.getElementById('compressed-diff');
    if (diff) {
        const diffLabel = diff.querySelector('label')
        diffLabel.addEventListener('click', ({ target }) =>
            !target.classList.contains('diff-button') && diff.classList.toggle('open')
        );
        const diffButton = document.createElement('button');
        diffButton.classList.add('diff-button');
        diffButton.addEventListener('click', () => diff.classList.toggle('show-diff'));
        diffLabel.appendChild(diffButton);
        window.openDiff = () => diff.classList.toggle('force-open');
    }

    // Question click handler
    const handleQuestionClick = (event) => {
        const promptInput = document.querySelector('.question-target textarea');
        promptInput.value = event.target.innerText;
        const inputEvent = new Event("input");
        Object.defineProperty(inputEvent, "target", { value: promptInput });
        promptInput.dispatchEvent(inputEvent);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    const qa_pairs = document.querySelector('.qa-pairs .table tbody');
    if (qa_pairs) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeName === 'TR') node.querySelector('td span').addEventListener('click', handleQuestionClick);
                });
                mutation.removedNodes.forEach((node) => {
                    if (node.nodeName === 'TR') node.querySelector('td span').removeEventListener('click', handleQuestionClick);
                });
            });
        });
        observer.observe(qa_pairs, { childList: true });
    }

    // Scroll to results on submit, scroll to top on clear
    document.getElementById('submit').addEventListener('click', () => {
        const results = document.getElementById('outputs');
        window.scrollTo({ top: results.offsetTop, behavior: 'smooth' });
    })
    document.getElementById('clear').addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

    // TODO: add synchronized resize for llm response textareas
}