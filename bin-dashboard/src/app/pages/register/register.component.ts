import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  templateUrl: './register.component.html',
  styleUrl: './register.component.css'
})
export class RegisterComponent {
    registerForm: FormGroup;
    errorMessage: string = "";
    
    constructor(
      private fb: FormBuilder,
      private http: HttpClient,
      private router: Router
    ) {
      this.registerForm = this.fb.group({
        email: ['', [Validators.required, Validators.email]],
        password: ['', Validators.required],
        confirmPassword: ['', Validators.required]
      });
    }
    onSubmit() {
      if (this.registerForm.invalid) return;
  
      const { email, password, confirmPassword } = this.registerForm.value;
  
      if (password !== confirmPassword) {
        this.errorMessage = 'Passwords do not match';
        return;
      }
  
      const userData = { email, password };
  
      this.http.post('api_link/api/user', userData).subscribe({   //change link here
        next: (response) => {
          console.log('Registration successful:', response);
          this.router.navigate(['/login']); // Redirect to login
        },
        error: (err) => {
          console.error('Registration failed:', err);
          this.errorMessage = 'Registration failed. Try again.';
        }
      });
    }
}
