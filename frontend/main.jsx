import React from 'react';
import { createRoot } from 'react-dom/client';
import CaptchaTester from './components/CaptchaTester';

const root = createRoot(document.getElementById('root'));
root.render(<CaptchaTester />);