() => {
    (() => {
        if (document.cookie.includes('session=')) return;
        const date = new Date(+new Date() + 10 * 365 * 24 * 60 * 60 * 1000);
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        document.cookie = `session=${Array(32).fill().map(() => chars.charAt(Math.floor(Math.random() * chars.length))).join('')}; expires=${date.toUTCString()}; path=/`;
    })();
    // Word count for textareas
    const wordCountHandler = (event) => {
        const words = event.target.value.match(/\w+/g)?.length ?? 0;
        const label = event.target.previousElementSibling;
        label.style = words ? `--word-count: " (${words} words)"` : '';
    }
    const wordCountFields = document.querySelectorAll('.word-count textarea');
    wordCountFields.forEach((t) => t.addEventListener('input', wordCountHandler));
    // Add listener to example prompt buttons to set wordcount on selection
    document.querySelectorAll('.gallery .gallery-item, button.clear').forEach((item) => {
        item.addEventListener('click', () => setTimeout(() => {
            wordCountFields.forEach((t) => wordCountHandler({ target: t }))
        }, 300));
    });
    // Question click handler
    const handleQuestionClick = (event) => {
        const promptCheckbox = document.querySelector('.ui-settings input');
        if (!promptCheckbox.checked) promptCheckbox.click();
        const promptInput = document.querySelector('.question-target input');
        promptInput.value = event.target.innerText;
        const inputEvent = new Event("input");
        Object.defineProperty(inputEvent, "target", { value: promptInput });
        promptInput.dispatchEvent(inputEvent);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
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
    observer.observe(document.querySelector('.qa-pairs .table tbody'), { childList: true });
}