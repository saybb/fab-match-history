// ==UserScript==
// @name         FaB History Scraper
// @version      1.0
// @description  Scrape match history data from a player's own page
// @author       Leon Schüßler
// @match        https://gem.fabtcg.com/profile/history/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const navigationDelay = 2000;
    let allEventData = JSON.parse(localStorage.getItem('allEventData')) || [];

    function scrapeEventData() {
        const currentPageLink = document.querySelector('.pagination-pages li.page-item.active');
        const currentPageIndex = currentPageLink ? parseInt(currentPageLink.textContent.trim()) : 1;

        const events = document.querySelectorAll('.event');

        events.forEach(event => {
            let eventName, eventDate, eventType, eventFormat, rated;
            
            // get relevant metadata about the event
            eventName = event.querySelector('.event__title')?.textContent.trim();

            const eventMeta = event.querySelectorAll('.event__meta-item');
            /**
             * 0: date
             * 1: store
             * 2: event type
             * 3: event format
             * 4: xp modifier
             * 5: is rated?
             */
            if (eventMeta) {
                eventDate = eventMeta[0].querySelector('span')?.textContent.trim()
                eventType = eventMeta[2].querySelector('span')?.textContent.trim()
                eventFormat = eventMeta[3].querySelector('span')?.textContent.trim()
                rated = eventMeta[5].querySelector('span')?.textContent.trim()
            }

            const eventResults = {
                eventName: eventName || 'Event name not found',
                eventDate: eventDate || 'Event date not found',
                eventType: eventType || 'Event type not found',
                eventFormat: eventFormat || 'Event format not found',
                rated: rated || 'Unknown', // Default to 'Unknown' if not found
                matches: []
            };
            
            // get match data
            const eventMatches = event.querySelectorAll('div.block-table table tbody tr:not(:first-child)');
            if (eventMatches) {
                eventMatches.forEach(match => {
                    const round = match.querySelector('td:nth-child(1)').textContent.trim();
                    const opponent = match.querySelector('td:nth-child(2)').textContent.trim();
                    const result = match.querySelector('td:nth-child(3)').textContent.trim();
                    eventResults.matches.push({ round, opponent, result });
                });

            }

            allEventData.push(eventResults);
        });

        console.log(`Scraped data from page ${currentPageIndex}.`);
        localStorage.setItem('allEventData', JSON.stringify(allEventData));
        navigateToNextPage(currentPageIndex);
    }

    function navigateToNextPage(currentPageIndex) {
        const currentPageLink = document.querySelector('.pagination-pages li.page-item.active');
        if (currentPageLink && currentPageLink.nextElementSibling) {
            const nextPageLink = currentPageLink.nextElementSibling.querySelector('a.page-link');
            if (nextPageLink) {
                console.log(`Navigating to page ${currentPageIndex + 1}...`);
                window.location.href = nextPageLink.href;
            } else {
                console.log('Reached the end of match history. Saving data...');
                saveDataToCSV();
                localStorage.removeItem('allEventData'); // Clear the stored data
            }
        } else {
            console.log('Reached the end of match history. Saving data...');
            saveDataToCSV();
            localStorage.removeItem('allEventData'); // Clear the stored data
        }
    }

    function saveDataToCSV() {
        // Convert allEventData to CSV format
        const csvRows = ['Event Name,Event Date,Event Type, Event Format,Rated,Round,Opponent,Result'];
        allEventData.forEach(event => {
            event.matches.forEach(match => {
                const row = [
                    `"${event.eventName}"`,
                    `"${event.eventDate}"`,
                    `"${event.eventType}"`,
                    `"${event.eventFormat}"`,
                    `"${event.rated}"`,
                    match.round,
                    `"${match.opponent}"`,
                    `"${match.result}"`,
                ].join(',');
                csvRows.push(row);
            });
        });

        const csvContent = csvRows.join('\n');

        // Create a Blob from the CSV content
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        // Create a link to download the file
        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.setAttribute('download', 'match_history.csv');
        document.body.appendChild(downloadLink);

        // Trigger download and remove the link
        downloadLink.click();
        document.body.removeChild(downloadLink);

        console.log('Scraped data saved to CSV.');
    }

    setTimeout(scrapeEventData, navigationDelay);
})();
